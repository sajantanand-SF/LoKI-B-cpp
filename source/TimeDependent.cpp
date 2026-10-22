/** \file
 *
 *  Time-dependent two-term Boltzmann solver. See LoKI-B/TimeDependent.h.
 */

#include "LoKI-B/TimeDependent.h"
#include "LoKI-B/Constant.h"
#include "LoKI-B/EedfUtilities.h"
#include "LoKI-B/Log.h"
#include "LoKI-B/StandardPaths.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace loki
{

namespace
{

double getDouble(const json_type &j)
{
    if (j.is_string())
        return std::stod(j.get<std::string>());
    return j.get<double>();
}

double getDouble(const json_type &cnf, const std::string &key, double def)
{
    return cnf.contains(key) ? getDouble(cnf.at(key)) : def;
}

bool getBool(const json_type &cnf, const std::string &key, bool def)
{
    if (!cnf.contains(key))
        return def;
    const auto &v = cnf.at(key);
    if (v.is_string())
        return v.get<std::string>() == "true";
    return v.get<bool>();
}

std::string getString(const json_type &cnf, const std::string &key, const std::string &def)
{
    return cnf.contains(key) ? cnf.at(key).get<std::string>() : def;
}

class ConstantWaveform : public Waveform
{
public:
    explicit ConstantWaveform(double v) : m_v(v) {}
    double operator()(double) const override { return m_v; }
private:
    double m_v;
};

/// MATLAB reducedFieldRamp: linear rise over riseTime, then constant
class RampWaveform : public Waveform
{
public:
    RampWaveform(double riseTime, double amplitude) : m_tau(riseTime), m_a(amplitude)
    {
        if (!(m_tau > 0))
            throw std::runtime_error("timeDependent: ramp riseTime must be positive.");
    }
    double operator()(double t) const override { return t <= 0 ? 0.0 : (t < m_tau ? m_a * t / m_tau : m_a); }
    std::vector<double> breakpoints(double t0, double t1) const override
    {
        return (m_tau > t0 && m_tau < t1) ? std::vector<double>{m_tau} : std::vector<double>{};
    }
private:
    double m_tau, m_a;
};

/// MATLAB reducedFieldPulse: amplitude*sqrt(t/tau)*exp(-t/tau)
class PulseWaveform : public Waveform
{
public:
    PulseWaveform(double charTime, double amplitude) : m_tau(charTime), m_a(amplitude) {}
    double operator()(double t) const override { return t <= 0 ? 0.0 : m_a * std::sqrt(t / m_tau) * std::exp(-t / m_tau); }
private:
    double m_tau, m_a;
};

/// offset + amplitude*sin(2 pi f t + phase)
class SinusoidWaveform : public Waveform
{
public:
    SinusoidWaveform(double amplitude, double frequency, double phase, double offset)
        : m_a(amplitude), m_f(frequency), m_phi(phase), m_off(offset)
    {
        if (!(m_f > 0))
            throw std::runtime_error("timeDependent: sinusoid frequency must be positive.");
    }
    double operator()(double t) const override { return m_off + m_a * std::sin(2 * Constant::pi * m_f * t + m_phi); }
    double period() const override { return 1.0 / m_f; }
    double peak(double, double) const override { return std::abs(m_off) + std::abs(m_a); }
    double rms(double, double) const override { return std::sqrt(m_off * m_off + 0.5 * m_a * m_a); }
private:
    double m_a, m_f, m_phi, m_off;
};

/// piecewise-linear table (t in s, E/N in Td), optionally periodic
class TableWaveform : public Waveform
{
public:
    TableWaveform(const std::filesystem::path &file, bool periodic) : m_periodic(periodic)
    {
        std::ifstream is(file);
        if (!is)
            throw std::runtime_error("timeDependent: cannot open waveform table '" + file.string() + "'.");
        std::string line;
        while (std::getline(is, line))
        {
            const auto p = line.find_first_not_of(" \t");
            if (p == std::string::npos || line[p] == '#' || line[p] == '%')
                continue;
            std::istringstream ls(line);
            double t, e;
            if (ls >> t >> e)
            {
                m_t.push_back(t);
                m_e.push_back(e);
            }
        }
        if (m_t.size() < 2)
            throw std::runtime_error("timeDependent: waveform table needs at least two rows.");
        for (std::size_t i = 1; i < m_t.size(); ++i)
            if (!(m_t[i] > m_t[i - 1]))
                throw std::runtime_error("timeDependent: waveform table times must be increasing.");
    }
    double operator()(double t) const override
    {
        if (m_periodic)
        {
            const double T = period();
            t = m_t.front() + std::fmod(std::fmod(t - m_t.front(), T) + T, T);
        }
        if (t <= m_t.front())
            return m_e.front();
        if (t >= m_t.back())
            return m_e.back();
        const auto it = std::upper_bound(m_t.begin(), m_t.end(), t);
        const std::size_t i = it - m_t.begin();
        const double w = (t - m_t[i - 1]) / (m_t[i] - m_t[i - 1]);
        return (1 - w) * m_e[i - 1] + w * m_e[i];
    }
    double period() const override { return m_periodic ? m_t.back() - m_t.front() : 0.0; }
    std::vector<double> breakpoints(double t0, double t1) const override
    {
        std::vector<double> b;
        if (!m_periodic)
        {
            for (double t : m_t)
                if (t > t0 && t < t1)
                    b.push_back(t);
            return b;
        }
        const double T = period();
        const double k0 = std::floor((t0 - m_t.front()) / T);
        for (double k = k0; m_t.front() + k * T < t1; k += 1)
            for (std::size_t i = 0; i + 1 < m_t.size(); ++i)
            {
                const double t = m_t[i] + k * T;
                if (t > t0 && t < t1)
                    b.push_back(t);
            }
        return b;
    }
private:
    std::vector<double> m_t, m_e;
    bool m_periodic;
};

} // namespace

double Waveform::peak(double t0, double t1) const
{
    double p = std::max(std::abs((*this)(t0)), std::abs((*this)(t1)));
    for (double t : breakpoints(t0, t1))
        p = std::max(p, std::abs((*this)(t)));
    const int n = 4000;
    for (int i = 1; i < n; ++i)
        p = std::max(p, std::abs((*this)(t0 + (t1 - t0) * i / n)));
    return p;
}

double Waveform::rms(double t0, double t1) const
{
    if (period() > 0)
    {
        t0 = 0;
        t1 = period();
    }
    const int n = 4000;
    double s = 0;
    for (int i = 0; i < n; ++i)
    {
        const double e = (*this)(t0 + (t1 - t0) * (i + 0.5) / n);
        s += e * e;
    }
    return std::sqrt(s / n);
}

std::unique_ptr<Waveform> Waveform::create(const json_type &cnf, const std::filesystem::path &basePath)
{
    if (cnf.is_number() || cnf.is_string())
        return std::make_unique<ConstantWaveform>(getDouble(cnf));
    const std::string type = cnf.at("type").get<std::string>();
    if (type == "constant")
        return std::make_unique<ConstantWaveform>(getDouble(cnf.at("value")));
    if (type == "ramp")
        return std::make_unique<RampWaveform>(getDouble(cnf.at("riseTime")), getDouble(cnf.at("amplitude")));
    if (type == "pulse")
        return std::make_unique<PulseWaveform>(getDouble(cnf.at("charTime")), getDouble(cnf.at("amplitude")));
    if (type == "sinusoid")
        return std::make_unique<SinusoidWaveform>(getDouble(cnf.at("amplitude")), getDouble(cnf.at("frequency")),
                                                  getDouble(cnf, "phase", 0.0), getDouble(cnf, "offset", 0.0));
    if (type == "table")
    {
        std::filesystem::path file = cnf.at("file").get<std::string>();
        if (file.is_relative() && !std::filesystem::exists(file))
            file = basePath.parent_path() / file;
        return std::make_unique<TableWaveform>(file, getBool(cnf, "periodic", false));
    }
    throw std::runtime_error("timeDependent: unknown reducedField type '" + type +
                             "' (constant, ramp, pulse, sinusoid, table).");
}

TimeDependentOptions::TimeDependentOptions(const json_type &cnf)
{
    const std::string ne = getString(cnf, "electronDensity", "swarm");
    if (ne == "swarm")
        electronDensity = ElectronDensity::swarm;
    else if (ne == "constant")
        electronDensity = ElectronDensity::constant;
    else
        throw std::runtime_error("timeDependent: electronDensity must be 'swarm' or 'constant', not '" + ne + "'.");

    if (cnf.contains("initialCondition"))
    {
        const auto &ic = cnf.at("initialCondition");
        const std::string type = getString(ic, "type", "steadyState");
        if (type == "steadyState")
        {
            initial = Initial::steadyState;
            if (ic.contains("reducedField"))
            {
                const auto &v = ic.at("reducedField");
                if (v.is_string() && (v.get<std::string>() == "initial" || v.get<std::string>() == "rms"))
                    initialReducedField = v.get<std::string>();
                else
                {
                    initialReducedField = "value";
                    initialReducedFieldValue = getDouble(v);
                }
            }
        }
        else if (type == "maxwellian")
        {
            initial = Initial::maxwellian;
            initialTemperature = getDouble(ic, "temperature", -1.0);
        }
        else
            throw std::runtime_error("timeDependent: initialCondition.type must be 'steadyState' or 'maxwellian'.");
    }
    gridFromPeakField = getBool(cnf, "gridFromPeakField", true);

    finalTime = getDouble(cnf.at("finalTime"));
    if (!(finalTime > 0))
        throw std::runtime_error("timeDependent: finalTime must be positive.");

    sampleTimes = {0.0, finalTime};
    if (cnf.contains("sampling"))
    {
        const auto &s = cnf.at("sampling");
        if (s.contains("times"))
            for (const auto &t : s.at("times"))
                sampleTimes.push_back(getDouble(t));
        if (s.contains("points"))
        {
            const int n = static_cast<int>(getDouble(s.at("points")));
            const std::string type = getString(s, "type", "linspace");
            const double first = getDouble(s, "first", finalTime / n);
            for (int i = 0; i < n; ++i)
            {
                const double x = n == 1 ? 1.0 : double(i) / (n - 1);
                if (type == "linspace")
                    sampleTimes.push_back(first + (finalTime - first) * x);
                else if (type == "logspace")
                    sampleTimes.push_back(first * std::pow(finalTime / first, x));
                else
                    throw std::runtime_error("timeDependent: sampling.type must be 'linspace' or 'logspace'.");
            }
        }
    }
    std::sort(sampleTimes.begin(), sampleTimes.end());
    sampleTimes.erase(std::remove_if(sampleTimes.begin(), sampleTimes.end(),
                                     [this](double t) { return t < 0 || t > finalTime * (1 + 1e-12); }),
                      sampleTimes.end());
    sampleTimes.erase(std::unique(sampleTimes.begin(), sampleTimes.end(),
                                  [this](double a, double b) { return std::abs(a - b) <= 1e-12 * finalTime; }),
                      sampleTimes.end());

    if (cnf.contains("integrator"))
    {
        const auto &ig = cnf.at("integrator");
        const std::string scheme = getString(ig, "scheme", "TR-BDF2");
        if (scheme != "TR-BDF2")
            throw std::runtime_error("timeDependent: only integrator.scheme 'TR-BDF2' is available.");
        rtol = getDouble(ig, "rtol", rtol);
        atol = getDouble(ig, "atol", atol);
        maxStep = getDouble(ig, "maxStep", maxStep);
        initialStep = getDouble(ig, "initialStep", initialStep);
        fixedStep = getDouble(ig, "fixedStep", fixedStep);
        const std::string nl = getString(ig, "nonLinear", "iterate");
        if (nl == "lag")
            lagNonLinear = true;
        else if (nl != "iterate")
            throw std::runtime_error("timeDependent: integrator.nonLinear must be 'iterate' or 'lag'.");
        maxNonLinearIter = static_cast<unsigned>(getDouble(ig, "maxNonLinearIterations", maxNonLinearIter));
        nonLinearTol = getDouble(ig, "nonLinearTolerance", nonLinearTol);
        maxSteps = static_cast<unsigned long>(getDouble(ig, "maxSteps", double(maxSteps)));
    }
}

TimeDependentBoltzmann::TimeDependentBoltzmann(ElectronKineticsBoltzmann &ek, WorkingConditions &wc,
                                               const json_type &cnf, const std::filesystem::path &basePath)
    : m_ek(ek), m_wc(wc), m_opt(cnf.at("electronKinetics").at("timeDependent")),
      m_waveform(Waveform::create(cnf.at("electronKinetics").at("timeDependent").at("reducedField"), basePath)),
      m_N(wc.gasDensity()), m_n(ek.grid().nCells()), m_hws(ek.grid().nCells())
{
    if (cnf.at("output").at("isOn").get<bool>())
        m_outFolder = std::filesystem::path(LOKIB_OUTPUT_DIR) / cnf.at("output").at("folder").get<std::string>() /
                      "timeDependent";
    checkSupported();
}

void TimeDependentBoltzmann::checkSupported() const
{
    if (m_wc.reducedExcFreqSI() != 0.0)
        throw std::runtime_error("timeDependent: excitationFrequency must be 0; the field oscillation is "
                                 "resolved in time through timeDependent.reducedField.");
    if (m_ek.eeOperator)
        throw std::runtime_error("timeDependent: electron-electron collisions are not supported yet.");
    if (m_ek.growthModelType == GrowthModelType::spatial &&
        m_ek.ionizationOperator.ionizationOperatorType != IonizationOperatorType::conservative)
        throw std::runtime_error("timeDependent: spatial growth is not available in the time-dependent mode; "
                                 "use growthModelType temporal (or conservative ionization).");
}

double TimeDependentBoltzmann::reducedField2(double t) const
{
    const double e = (*m_waveform)(t) * SI::Townsend;
    return e * e;
}

double TimeDependentBoltzmann::norm(const Vector &f) const
{
    return f.dot(m_numberWeights);
}

double TimeDependentBoltzmann::ciEff(const Vector &f) const
{
    return m_nonConservative ? f.dot(m_coefsCI) / norm(f) : 0.0;
}

void TimeDependentBoltzmann::assembleFixedOperators()
{
    const Grid &grid = m_ek.grid();
    m_n = grid.nCells();
    m_sqrtU = grid.getCells().cwiseSqrt();
    m_W = m_sqrtU / SI::gamma;
    m_numberWeights = m_sqrtU.cwiseProduct(grid.duCells());

    auto &ion = m_ek.ionizationOperator;
    auto &att = m_ek.attachmentOperator;
    m_nonConservative = ion.includeNonConservativeIonization || att.includeNonConservativeAttachment;
    if (m_nonConservative && m_ek.growthModelType != GrowthModelType::temporal)
        throw std::runtime_error("timeDependent: non-conservative ionization/attachment needs growthModelType temporal.");

    auto addIfSized = [this](Matrix &dst, const Matrix &src) {
        if (src.rows() == m_n && src.cols() == m_n)
            dst += src;
    };
    m_A0 = Matrix(m_ek.elasticMatrix);
    addIfSized(m_A0, m_ek.inelasticOperator.inelasticMatrix);
    if (m_nonConservative)
    {
        addIfSized(m_A0, ion.ionizationMatrix);
        if (att.attachmentMatrix.rows() == m_n)
            m_A0 += Matrix(att.attachmentMatrix);
        Matrix nc = Matrix::Zero(m_n, m_n);
        addIfSized(nc, ion.ionizationMatrix);
        if (att.attachmentMatrix.rows() == m_n)
            nc += Matrix(att.attachmentMatrix);
        // as in solveTemporalGrowthMatrix: CIEff = <nu_eff>/N = coefsCI . eedf
        m_coefsCI = SI::gamma * (grid.duCells().transpose() * nc).transpose();
    }
    else
    {
        addIfSized(m_A0, ion.ionConservativeMatrix);
        addIfSized(m_A0, att.attachmentConservativeMatrix);
        m_coefsCI = Vector::Zero(m_n);
    }
    if (m_ek.carOperator)
        m_A0 += Matrix(m_ek.CARMatrix);

    m_F = m_ek.fieldMatrix;  // tridiagonal pattern
    m_Fc = std::numeric_limits<double>::quiet_NaN();

    const auto bw = calculateBandwidth(m_A0);
    const auto lower = std::min<Matrix::Index>(bw.first, -1), upper = std::max<Matrix::Index>(bw.second, 1);
    m_solver = (lower == -1 && upper == 1) ? Solver::tdma : (lower == -1 ? Solver::hessenberg : Solver::lu);
    m_work.resize(m_n, m_n);
    m_hws.resize(m_n);
    Log<Message>::Notify("timeDependent: ", m_n, " cells, uMax = ", grid.uMax(), " eV, bandwidth [", lower, ",", upper,
                         "], solver ",
                         m_solver == Solver::tdma ? "TDMA" : (m_solver == Solver::hessenberg ? "Hessenberg" : "LU"),
                         ", growth ", m_nonConservative ? "non-conservative (temporal)" : "none (conservative)",
                         ", electronDensity ",
                         m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm ? "swarm" : "constant");
}

void TimeDependentBoltzmann::evaluateField(double c)
{
    const double cEff =
        (m_nonConservative && m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm) ? c : 0.0;
    if (!(cEff == m_Fc))
    {
        m_ek.fieldOperator.evaluate(m_ek.grid(), m_ek.mixture.collision_data().totalCrossSection(), 1.0, 0.0, cEff, m_F);
        m_Fc = cEff;
    }
}

Vector TimeDependentBoltzmann::applyNA(const Vector &f, double EoN2, double c) const
{
    // evaluateField is non-const (it caches F); callers make sure F matches c
    Vector r = m_A0 * f + EoN2 * (m_F * f);
    if (m_nonConservative)
        r -= (c / SI::gamma) * m_sqrtU.cwiseProduct(f);
    return m_N * r;
}

void TimeDependentBoltzmann::solveSystem(double a, double EoN2, double c, Vector &x)
{
    evaluateField(c);
    const double s = -a * m_N;
    m_work.noalias() = s * m_A0;
    for (int k = 0; k < m_F.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(m_F, k); it; ++it)
            m_work(it.row(), it.col()) += s * EoN2 * it.value();
    for (Grid::Index k = 0; k < m_n; ++k)
    {
        m_work(k, k) += m_W[k];
        if (m_nonConservative)
            m_work(k, k) += a * m_N * (c / SI::gamma) * m_sqrtU[k];
    }
    if (m_solver == Solver::lu)
        m_lu.compute(m_work);
    resolve(x);
}

void TimeDependentBoltzmann::resolve(Vector &x)
{
    ++m_diag.solves;
    switch (m_solver)
    {
    case Solver::tdma:
        LinAlg::solveTDMA(m_work, x);
        break;
    case Solver::hessenberg:
        LinAlg::hessenberg(m_work.data(), x.data(), m_n, m_hws);
        break;
    case Solver::lu:
        x = m_lu.solve(x);
        break;
    }
}

void TimeDependentBoltzmann::stage(double a, double EoN2, const Vector &rhs, double &c, Vector &x)
{
    if (!m_nonConservative || m_opt.lagNonLinear)
    {
        x = rhs;
        solveSystem(a, EoN2, c, x);
        return;
    }
    for (unsigned it = 0;; ++it)
    {
        x = rhs;
        solveSystem(a, EoN2, c, x);
        const double cNew = ciEff(x);
        if (std::abs(cNew - c) <= m_opt.nonLinearTol * std::abs(cNew) || cNew == c)
            break;
        if (it + 1 >= m_opt.maxNonLinearIter)
        {
            Log<Message>::Warning("timeDependent: CIEff iteration did not converge at t = ", m_t, " s (|dc/c| = ",
                                  std::abs(cNew - c) / std::abs(cNew), ").");
            break;
        }
        c = cNew;
    }
}

bool TimeDependentBoltzmann::step(double h, double &errNorm)
{
    static const double g = 2.0 - std::sqrt(2.0);
    static const double d = (1.0 - g) / (2.0 - g);
    static const double k = (-3.0 * g * g + 4.0 * g - 2.0) / (12.0 * (2.0 - g));

    evaluateField(m_c);
    const Vector zn = applyNA(m_f, reducedField2(m_t), m_c).cwiseQuotient(m_W);

    // stage 1: trapezoidal rule over [t, t + g h]
    const Vector rhs1 = m_W.cwiseProduct(m_f + (0.5 * g * h) * zn);
    double cg = m_c;
    Vector fg(m_n);
    stage(0.5 * g * h, reducedField2(m_t + g * h), rhs1, cg, fg);
    const Vector zg = (2.0 / (g * h)) * (fg - m_f) - zn;

    // stage 2: BDF2 over [t, t + g h, t + h]
    const Vector rhs2 = m_W.cwiseProduct(fg / (g * (2.0 - g)) - ((1.0 - g) * (1.0 - g) / (g * (2.0 - g))) * m_f);
    double c1 = cg;
    Vector f1(m_n);
    stage(d * h, reducedField2(m_t + h), rhs2, c1, f1);

    errNorm = 0.0;
    if (m_opt.fixedStep <= 0)
    {
        const Vector z1 = (f1 - rhs2.cwiseQuotient(m_W)) / (d * h);
        // local error estimate (Bank et al. 1985; Hosea and Shampine 1996), filtered by the stage-2 matrix
        Vector est = m_W.cwiseProduct((2.0 * k * h) * (zn / g - zg / (g * (1.0 - g)) + z1 / (1.0 - g)));
        resolve(est);
        const double fmax = f1.cwiseAbs().maxCoeff();
        for (Grid::Index i = 0; i < m_n; ++i)
        {
            const double w = m_opt.rtol * std::max(std::abs(f1[i]), std::abs(m_f[i])) + m_opt.atol * fmax;
            errNorm = std::max(errNorm, std::abs(est[i]) / w);
        }
        if (!(errNorm <= 1.0))
            return false;
    }

    const double nrm = norm(f1);
    m_diag.maxNormDrift = std::max(m_diag.maxNormDrift, std::abs(nrm - 1.0));
    m_f = f1 / nrm;
    m_c = ciEff(m_f);
    m_t += h;
    m_diag.lastStep = h;
    m_diag.minEedf = std::min(m_diag.minEedf, m_f.minCoeff());
    return true;
}

void TimeDependentBoltzmann::initialize()
{
    const Grid &grid = m_ek.grid();
    if (m_opt.gridFromPeakField && grid.smartGrid())
    {
        const double peak = m_waveform->peak(0.0, m_opt.finalTime);
        Log<Message>::Notify("timeDependent: choosing the energy grid with the smart grid at the peak field ", peak,
                             " Td.");
        m_wc.updateReducedField(peak);
        m_ek.solveSmartGrid();
    }
    assembleFixedOperators();

    if (m_opt.initial == TimeDependentOptions::Initial::steadyState)
    {
        double e0 = (*m_waveform)(0.0);
        if (m_opt.initialReducedField == "rms")
            e0 = m_waveform->rms(0.0, m_opt.finalTime);
        else if (m_opt.initialReducedField == "value")
            e0 = m_opt.initialReducedFieldValue;
        Log<Message>::Notify("timeDependent: initial state is the steady solution at ", e0, " Td.");
        m_wc.updateReducedField(e0);
        m_ek.solveSingle();
        m_f = m_ek.eedf;
    }
    else
    {
        const double T = m_opt.initialTemperature > 0 ? m_opt.initialTemperature
                                                       : Constant::kBeV * m_wc.gasTemperature();
        Log<Message>::Notify("timeDependent: initial state is a Maxwellian at ", T, " eV.");
        m_f = (-grid.getCells() / T).array().exp();
    }
    m_f /= norm(m_f);
    m_c = ciEff(m_f);
    m_t = 0.0;
    m_nextSample = 0;
    m_diag = Diagnostics();
    m_diag.minEedf = m_f.minCoeff();

    if (m_opt.fixedStep > 0)
        m_h = m_opt.fixedStep;
    else if (m_opt.initialStep > 0)
        m_h = m_opt.initialStep;
    else
    {
        evaluateField(m_c);
        const Vector z = applyNA(m_f, reducedField2(0.0), m_c).cwiseQuotient(m_W);
        const double fmax = m_f.cwiseAbs().maxCoeff();
        double r = 0;
        for (Grid::Index i = 0; i < m_n; ++i)
            r = std::max(r, std::abs(z[i]) / (std::abs(m_f[i]) + m_opt.atol * fmax));
        m_h = r > 0 ? 1e-2 * std::cbrt(m_opt.rtol) / r : 1e-3 * m_opt.finalTime;
        m_h = std::min(m_h, 0.1 * m_opt.finalTime);
    }
}

void TimeDependentBoltzmann::advance(double tEnd)
{
    std::vector<double> stops = m_waveform->breakpoints(m_t, tEnd);
    for (double t : m_opt.sampleTimes)
        if (t > m_t && t < tEnd)
            stops.push_back(t);
    stops.push_back(tEnd);
    std::sort(stops.begin(), stops.end());
    std::size_t iStop = 0;
    const double eps = 1e-12 * std::max(tEnd, m_opt.finalTime);

    while (m_t < tEnd - eps)
    {
        while (stops[iStop] <= m_t + eps)
            ++iStop;
        const double target = stops[iStop];
        double hTrial = m_opt.fixedStep > 0 ? m_opt.fixedStep : m_h;
        if (m_opt.maxStep > 0)
            hTrial = std::min(hTrial, m_opt.maxStep);
        double h = hTrial;
        if (m_t + h >= target - eps)
            h = target - m_t;
        else if (m_t + 1.5 * h > target)
            h = 0.5 * (target - m_t);

        double err = 0;
        const bool ok = step(h, err);
        ++m_diag.steps;
        if (m_diag.steps > m_opt.maxSteps)
            throw std::runtime_error("timeDependent: maxSteps exceeded.");
        if (m_opt.fixedStep <= 0)
        {
            const double fac = std::clamp(err > 0 ? 0.9 * std::pow(err, -1.0 / 3.0) : 5.0, 0.2, 5.0);
            if (ok)
                m_h = h < hTrial ? std::max(hTrial, h * fac) : h * fac;  // a shortened landing step keeps hTrial
            else
            {
                ++m_diag.rejected;
                m_h = h * fac;
                if (m_h < 1e-14 * std::max(m_t, m_opt.finalTime))
                    throw std::runtime_error("timeDependent: step size underflow.");
            }
        }
        if (ok)
            while (m_nextSample < m_opt.sampleTimes.size() &&
                   std::abs(m_opt.sampleTimes[m_nextSample] - m_t) <= eps)
            {
                sample();
                ++m_nextSample;
            }
    }
}

void TimeDependentBoltzmann::openOutput()
{
    if (m_outFolder.empty())
        return;
    std::filesystem::create_directories(m_outFolder);
    m_timeSeries.open(m_outFolder / "timeSeries.txt");
    m_timeSeries << "# time-dependent LoKI-B: N = " << m_N << " m^-3, " << m_n << " cells, uMax = " << m_ek.grid().uMax()
                 << " eV\n";
    m_timeSeries << "# columns: t(s) E/N(Td) meanEnergy(eV) Te(eV) redMob(1/(msV)) redDiff(1/(ms)) "
                    "redTownsend(m2) CIEff(m3/s) Pfield Pelastic Pcar Pinelastic Psuperelastic PeDensGrowth "
                    "Pbalance(eVm3/s) depsdt(eV/s) powerResidual minEedf steps rejected solves lastStep(s)\n";
    m_timeSeries << std::scientific << std::setprecision(10);
    m_rates.open(m_outFolder / "rateCoefficients_t.txt");
    m_rates << "# columns: t(s) E/N(Td), then for each collision the inelastic and superelastic rate coefficients "
               "(m3/s)\n# collisions:";
    m_ek.mixture.collision_data().evaluateRateCoefficients(m_ek.grid(), m_f);
    for (const auto &rc : m_ek.mixture.collision_data().rateCoefficients())
        m_rates << " [" << *rc.collision << "]";
    m_rates << '\n' << std::scientific << std::setprecision(10);
}

void TimeDependentBoltzmann::sample()
{
    const double EoN = (*m_waveform)(m_t);
    const Grid &grid = m_ek.grid();
    m_wc.updateReducedField(EoN);
    m_ek.eedf = m_f;
    m_ek.CIEff = m_c;
    const double cEff =
        (m_nonConservative && m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm) ? m_c : 0.0;
    m_ek.fieldOperator.evaluate(grid, m_ek.mixture.collision_data().totalCrossSection(), 1.0, 0.0, cEff);
    m_ek.evaluatePower();
    m_ek.mixture.collision_data().evaluateRateCoefficients(grid, m_ek.eedf);
    m_ek.evaluateSwarmParameters();

    evaluateField(m_c);
    const Vector z = applyNA(m_f, reducedField2(m_t), m_c).cwiseQuotient(m_W);
    const double depsdt = grid.getCells().array().pow(1.5).matrix().cwiseProduct(grid.duCells()).dot(z);
    const Power &p = m_ek.power;
    const double residual = (depsdt - m_N * p.balance) / (m_N * p.reference);

    if (m_timeSeries.is_open())
    {
        const SwarmParameters &s = m_ek.swarmParameters;
        m_timeSeries << m_t << ' ' << EoN << ' ' << s.meanEnergy << ' ' << s.Te << ' ' << s.redMobCoeff << ' '
                     << s.redDiffCoeff << ' ' << s.redTownsendCoeff << ' ' << m_c << ' ' << p.field << ' '
                     << p.elasticNet << ' ' << p.carNet << ' ' << p.inelastic << ' ' << p.superelastic << ' '
                     << p.eDensGrowth << ' ' << p.balance << ' ' << depsdt << ' ' << residual << ' '
                     << m_f.minCoeff() << ' ' << m_diag.steps << ' ' << m_diag.rejected << ' ' << m_diag.solves << ' '
                     << m_diag.lastStep << '\n';
        m_timeSeries.flush();
        m_rates << m_t << ' ' << EoN;
        for (const auto &rc : m_ek.mixture.collision_data().rateCoefficients())
            m_rates << ' ' << rc.inelastic << ' ' << rc.superelastic;
        m_rates << '\n';
    }
    m_eedfTimes.push_back(m_t);
    m_eedfFields.push_back(EoN);
    m_eedfSamples.push_back(m_f);
}

void TimeDependentBoltzmann::emitState(const std::string &jobFolder)
{
    const Grid &grid = m_ek.grid();
    const double EoN = (*m_waveform)(m_t);
    m_wc.setCurrentJobFolder(jobFolder);
    m_wc.updateReducedField(EoN);
    m_ek.eedf = m_f;
    m_ek.CIEff = m_c;
    const double cEff =
        (m_nonConservative && m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm) ? m_c : 0.0;
    m_ek.fieldOperator.evaluate(grid, m_ek.mixture.collision_data().totalCrossSection(), 1.0, 0.0, cEff);
    m_ek.evaluatePower();
    m_ek.mixture.collision_data().evaluateRateCoefficients(grid, m_ek.eedf);
    m_ek.evaluateSwarmParameters();
    m_ek.evaluateFirstAnisotropy();
    m_ek.obtainedNewEedf.emit(grid, m_ek.eedf, m_wc, m_ek.power, m_ek.mixture.collision_data(), m_ek.swarmParameters,
                              &m_ek.firstAnisotropy);
}

void TimeDependentBoltzmann::run()
{
    const auto begin = std::chrono::steady_clock::now();
    initialize();
    openOutput();
    if (!m_opt.sampleTimes.empty() && m_opt.sampleTimes.front() <= 0.0)
    {
        sample();
        ++m_nextSample;
    }
    emitState("td_initial");
    advance(m_opt.finalTime);
    emitState("td_final");
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::ostringstream summary;
    summary << "finalTime " << m_opt.finalTime << " s\nsteps " << m_diag.steps << "\nrejected " << m_diag.rejected
            << "\nsolves " << m_diag.solves << "\nmaxNormDrift " << m_diag.maxNormDrift << "\nminEedf "
            << m_diag.minEedf << "\ncells " << m_n << "\nuMax " << m_ek.grid().uMax() << "\nwallTime " << elapsed
            << " s\n";
    Log<Message>::Notify("timeDependent finished:\n", summary.str());
    if (m_outFolder.empty())
        return;
    std::ofstream(m_outFolder / "summary.txt") << summary.str();
    std::ofstream os(m_outFolder / "eedf_t.txt");
    os << "# EEDF (eV^-3/2) at the sampled times; column 1: energy (eV), then one column per time\n# t(s):";
    os << std::scientific << std::setprecision(10);
    for (double t : m_eedfTimes)
        os << ' ' << t;
    os << "\n# E/N(Td):";
    for (double e : m_eedfFields)
        os << ' ' << e;
    os << '\n';
    for (Grid::Index i = 0; i < m_n; ++i)
    {
        os << m_ek.grid().getCell(i);
        for (const auto &f : m_eedfSamples)
            os << ' ' << f[i];
        os << '\n';
    }
}

} // namespace loki
