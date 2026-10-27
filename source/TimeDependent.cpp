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
    bool sinusoid(double &amplitude, double &omega, double &phase, double &offset) const override
    {
        amplitude = m_a;
        omega = 2 * Constant::pi * m_f;
        phase = m_phi;
        offset = m_off;
        return true;
    }
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

/// replace the field power (and recompute the balance) after ElectronKineticsBoltzmann::evaluatePower
void setFieldPower(Power &pw, double field)
{
    pw.field = field;
    pw.balance = pw.field + pw.elasticNet + pw.carNet + pw.inelastic + pw.superelastic + pw.eDensGrowth +
                 pw.electronElectronNet;
    pw.reference = std::max(pw.field, 0.0) + std::max(pw.elasticGain, 0.0) + std::max(pw.carGain, 0.0) +
                   std::max(pw.superelastic, 0.0) + std::max(pw.electronElectronGain, 0.0);
    pw.relativeBalance = pw.reference > 0 ? std::abs(pw.balance) / pw.reference : 0.0;
}

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
    else if (ne == "external")
        electronDensity = ElectronDensity::external;
    else
        throw std::runtime_error("timeDependent: electronDensity must be 'swarm', 'constant' or 'external', not '" + ne +
                                 "'.");
    evolveElectronDensity = getBool(cnf, "evolveElectronDensity", false);

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
    const std::string an = getString(cnf, "anisotropy", "quasiStatic");
    if (an == "quasiStatic")
        anisotropy = Anisotropy::quasiStatic;
    else if (an == "dynamic")
        anisotropy = Anisotropy::dynamic;
    else
        throw std::runtime_error("timeDependent: anisotropy must be 'quasiStatic' or 'dynamic', not '" + an + "'.");

    if (cnf.contains("periodic"))
    {
        const auto &pc = cnf.at("periodic");
        periodic = getBool(pc, "isOn", true);
        periodicTol = getDouble(pc, "tol", periodicTol);
        periodicAtol = getDouble(pc, "atol", periodicAtol);
        minPeriods = static_cast<unsigned>(getDouble(pc, "minPeriods", minPeriods));
        maxPeriods = static_cast<unsigned>(getDouble(pc, "maxPeriods", maxPeriods));
        averagePeriods = static_cast<unsigned>(getDouble(pc, "averagePeriods", averagePeriods));
        samplesPerPeriod = static_cast<unsigned>(getDouble(pc, "samplesPerPeriod", samplesPerPeriod));
        // dynamic f1 oscillates at omega; TR-BDF2 resolves the forced response to ~(omega h)^2/12, so it needs finer
        // steps than quasi-static f1 (V7: 32 steps per period bias the heating by ~1e-3, 128 by < 1e-4)
        stepsPerPeriod = static_cast<unsigned>(
            getDouble(pc, "stepsPerPeriod", anisotropy == Anisotropy::dynamic ? 128u : stepsPerPeriod));
        freezeTol = getDouble(pc, "freezeTol", freezeTol);
        if (averagePeriods < 1 || samplesPerPeriod < 1 || stepsPerPeriod < 1 || minPeriods < 2)
            throw std::runtime_error("timeDependent.periodic: averagePeriods, samplesPerPeriod, stepsPerPeriod >= 1 and "
                                     "minPeriods >= 2.");
    }
    if (periodic)
        finalTime = getDouble(cnf, "finalTime", 0.0);  // not used to stop the run
    else
    {
        finalTime = getDouble(cnf.at("finalTime"));
        if (!(finalTime > 0))
            throw std::runtime_error("timeDependent: finalTime must be positive.");
    }

    sampleTimes = {0.0, finalTime};
    if (!periodic && cnf.contains("sampling"))
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
            nonLinear = NonLinear::lag;
        else if (nl == "extrapolate")
            nonLinear = NonLinear::extrapolate;
        else if (nl != "iterate")
            throw std::runtime_error("timeDependent: integrator.nonLinear must be 'iterate', 'lag' or 'extrapolate'.");
        maxNonLinearIter = static_cast<unsigned>(getDouble(ig, "maxNonLinearIterations", maxNonLinearIter));
        nonLinearTol = getDouble(ig, "nonLinearTolerance", nonLinearTol);
        eeTol = getDouble(ig, "eeTolerance", eeTol);
        maxSteps = static_cast<unsigned long>(getDouble(ig, "maxSteps", double(maxSteps)));
        const std::string ls = getString(ig, "linearSolver", "auto");
        if (ls == "lu")
            linearSolver = LinearSolver::lu;
        else if (ls == "sparse")
            linearSolver = LinearSolver::sparse;
        else if (ls != "auto")
            throw std::runtime_error("timeDependent: integrator.linearSolver must be 'auto', 'lu' or 'sparse'.");
        krylovTol = getDouble(ig, "krylovTolerance", krylovTol);
        krylovMaxIter = static_cast<unsigned>(getDouble(ig, "krylovMaxIterations", krylovMaxIter));
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
    if (m_opt.periodic && !(m_waveform->period() > 0))
        throw std::runtime_error("timeDependent: periodic mode needs a periodic reducedField (sinusoid, or table with "
                                 "periodic: true).");
    if (m_opt.periodic && m_opt.maxStep <= 0 && m_opt.fixedStep <= 0)
        m_opt.maxStep = m_waveform->period() / m_opt.stepsPerPeriod;
}

void TimeDependentBoltzmann::checkSupported() const
{
    if (m_wc.reducedExcFreqSI() != 0.0)
        throw std::runtime_error("timeDependent: excitationFrequency must be 0; the field oscillation is "
                                 "resolved in time through timeDependent.reducedField.");
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
    m_ee = static_cast<bool>(m_ek.eeOperator);
    if (m_ee)
        m_ek.eeOperator->initialize(grid);  // the a matrix of the e-e operator for the final grid
    m_sqrtU = grid.getCells().cwiseSqrt();
    m_W = m_sqrtU / SI::gamma;
    m_numberWeights = m_sqrtU.cwiseProduct(grid.duCells());
    m_uDu = grid.getCells().cwiseProduct(grid.duCells());
    m_u32du = m_uDu.cwiseProduct(m_sqrtU);
    m_dynF1 = m_opt.anisotropy == TimeDependentOptions::Anisotropy::dynamic;
    m_uNode = grid.getNodes();
    m_sqrtUNode = m_uNode.cwiseSqrt();
    m_dc = grid.isUniform() ? Vector(Vector::Constant(m_n, grid.du())) : Vector(grid.duCells());
    m_dn = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        m_dn[j] = grid.isUniform() ? grid.du() : grid.duNode(j);

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
    m_solver = (lower == -1 && upper == 1)
                   ? Solver::tdma
                   : (lower == -1 ? Solver::hessenberg
                                  : (m_opt.linearSolver == TimeDependentOptions::LinearSolver::lu ? Solver::lu
                                                                                                  : Solver::krylov));
    if (m_opt.linearSolver == TimeDependentOptions::LinearSolver::sparse && m_solver != Solver::tdma)
    {
        m_denseSolver = m_solver == Solver::hessenberg ? Solver::hessenberg : Solver::krylov;
        m_solver = Solver::sparse;
        m_A0lo = Vector::Zero(m_n);
        m_A0d = m_A0.diagonal();
        m_A0up = Vector::Zero(m_n);
        std::vector<Eigen::Triplet<double>> rest;
        for (Grid::Index j = 0; j < m_n; ++j)
            for (Grid::Index i = 0; i < m_n; ++i)
            {
                const double v = m_A0(i, j);
                if (i == j + 1)
                    m_A0lo[i] = v;
                else if (i + 1 == j)
                    m_A0up[i] = v;
                else if (i != j && v != 0.0)
                    rest.emplace_back(i, j, v);
            }
        m_stage.R.resize(m_n, m_n);
        m_stage.R.setFromTriplets(rest.begin(), rest.end());
        m_stage.R.makeCompressed();
        Log<Message>::Notify("timeDependent: sparse stage matrices, ", rest.size(), " entries off the tridiagonal (",
                             100.0 * double(rest.size()) / (double(m_n) * double(m_n)), " % of N^2)");
    }
    m_work.resize(m_n, m_n);
    m_hws.resize(m_n);
    Log<Message>::Notify("timeDependent: ", m_n, " cells, uMax = ", grid.uMax(), " eV, bandwidth [", lower, ",", upper,
                         "], solver ",
                         m_solver == Solver::tdma
                             ? "TDMA"
                             : (m_solver == Solver::hessenberg
                                    ? "Hessenberg"
                                    : (m_solver == Solver::lu ? "LU"
                                                              : (m_solver == Solver::sparse
                                                                     ? "GMRES (sparse, tridiagonal-preconditioned)"
                                                                     : "GMRES (Hessenberg-preconditioned)"))),
                         ", growth ", m_nonConservative ? "non-conservative (temporal)" : "none (conservative)",
                         ", electronDensity ",
                         m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm
                             ? "swarm"
                             : (m_opt.electronDensity == TimeDependentOptions::ElectronDensity::constant ? "constant"
                                                                                                         : "external"),
                         m_ee ? ", e-e collisions on" : "", m_dynF1 ? ", dynamic f1" : ", quasi-static f1");
}

double TimeDependentBoltzmann::fieldGrowthTerm(double c) const
{
    using ED = TimeDependentOptions::ElectronDensity;
    switch (m_opt.electronDensity)
    {
    case ED::swarm:
        return m_nonConservative ? c : 0.0;
    case ED::constant:
        return 0.0;
    case ED::external:
        return m_dlnneDt / m_N;
    }
    return 0.0;
}

void TimeDependentBoltzmann::evaluateField(double c)
{
    const double cEff = fieldGrowthTerm(c);
    if (!(cEff == m_Fc))
    {
        m_ek.fieldOperator.evaluate(m_ek.grid(), m_ek.mixture.collision_data().totalCrossSection(), 1.0, 0.0, cEff, m_F);
        m_Fc = cEff;
    }
}

TimeDependentBoltzmann::NonLinearState TimeDependentBoltzmann::nonLinearState(const Vector &f) const
{
    NonLinearState nl;
    nl.c = ciEff(f);
    if (m_ee)
    {
        const Vector fn = f / norm(f);
        m_ek.eeOperator->update_g_ee_AB(m_ek.grid(), fn, m_ne, m_N);
        nl.A = m_ek.eeOperator->A();
        nl.B = m_ek.eeOperator->B();
    }
    return nl;
}

Vector TimeDependentBoltzmann::applyNA(const Vector &f, double EoN2, const NonLinearState &nl) const
{
    // evaluateField is non-const (it caches F); callers make sure F matches nl.c
    Vector r = m_solver == Solver::sparse ? Vector(sparseA0Product(f) + EoN2 * (m_F * f))
                                          : Vector(m_A0 * f + EoN2 * (m_F * f));
    if (m_nonConservative)
        r -= (nl.c / SI::gamma) * m_sqrtU.cwiseProduct(f);
    if (m_ee)
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            r[k] -= (nl.A[k] + nl.B[k]) * f[k];
            if (k > 0)
                r[k] += nl.A[k - 1] * f[k - 1];
            if (k < m_n - 1)
                r[k] += nl.B[k + 1] * f[k + 1];
        }
    return m_N * r;
}

void TimeDependentBoltzmann::solveSystem(double a, double EoN2, const NonLinearState &nl, Vector &x)
{
    evaluateField(nl.c);
    const double s = -a * m_N;
    m_luCurrent = false;
    if (m_solver == Solver::sparse)
    {
        // the same terms as below, into the tridiagonal part of m_stage (kept separate from the dense assembly so
        // that its floating-point operations, FMA contractions included, stay as they were)
        m_stage.setTridiagonal(s, m_A0lo, m_A0d, m_A0up);
        for (int k = 0; k < m_F.outerSize(); ++k)
            for (SparseMatrix::InnerIterator it(m_F, k); it; ++it)
                m_stage.add(it.row(), it.col(), s * EoN2 * it.value());
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            m_stage.d[k] += m_W[k];
            if (m_nonConservative)
                m_stage.d[k] += a * m_N * (nl.c / SI::gamma) * m_sqrtU[k];
            if (m_ee)
            {
                m_stage.d[k] -= s * (nl.A[k] + nl.B[k]);
                if (k > 0)
                    m_stage.lo[k] += s * nl.A[k - 1];
                if (k < m_n - 1)
                    m_stage.up[k] += s * nl.B[k + 1];
            }
        }
        m_stage.factor();
        m_denseCurrent = false;
        resolve(x);
        return;
    }
    m_work.noalias() = s * m_A0;
    for (int k = 0; k < m_F.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(m_F, k); it; ++it)
            m_work(it.row(), it.col()) += s * EoN2 * it.value();
    for (Grid::Index k = 0; k < m_n; ++k)
    {
        m_work(k, k) += m_W[k];
        if (m_nonConservative)
            m_work(k, k) += a * m_N * (nl.c / SI::gamma) * m_sqrtU[k];
        if (m_ee)
        {
            // same discretization as ElectronElectronOperator::discretizeTerm
            m_work(k, k) -= s * (nl.A[k] + nl.B[k]);
            if (k > 0)
                m_work(k, k - 1) += s * nl.A[k - 1];
            if (k < m_n - 1)
                m_work(k, k + 1) += s * nl.B[k + 1];
        }
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
    case Solver::krylov:
        solveKrylov(x);
        break;
    case Solver::sparse:
        solveSparse(x);
        break;
    }
}

namespace
{

/// GMRES from x0 = 0 for M x = b (x overwrites b), right-preconditioned: apply(z) returns M z, precondition(z) sets
/// z <- P^-1 z. Returns the number of iterations, or 0 if not converged after maxIter (b is then unchanged).
template <class Apply, class Precondition>
unsigned gmres(Eigen::Index n, const Apply &apply, const Precondition &precondition, Vector &b, double tol,
               unsigned maxIter, Matrix &V, Matrix &Z)
{
    const double bnorm = b.norm();
    if (bnorm == 0.0)
        return 1;
    if (!std::isfinite(bnorm))
        return 0;
    // right preconditioning: M P^-1 v = b, x = P^-1 v; Givens rotations on the small Hessenberg matrix H
    const unsigned m = std::max(1u, maxIter);
    V.resize(n, m + 1);
    Z.resize(n, m);
    Matrix H = Matrix::Zero(m + 1, m);
    Vector cs(m), sn(m), g = Vector::Zero(m + 1);
    g[0] = bnorm;
    V.col(0) = b / bnorm;
    unsigned j = 0;
    while (j < m)
    {
        Vector z = V.col(j);
        precondition(z);
        Z.col(j) = z;
        Vector w = apply(z);
        for (unsigned i = 0; i <= j; ++i)  // modified Gram-Schmidt
        {
            H(i, j) = V.col(i).dot(w);
            w -= H(i, j) * V.col(i);
        }
        H(j + 1, j) = w.norm();
        if (H(j + 1, j) > 0.0)
            V.col(j + 1) = w / H(j + 1, j);
        for (unsigned i = 0; i < j; ++i)
        {
            const double t = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
            H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
            H(i, j) = t;
        }
        const double r = std::hypot(H(j, j), H(j + 1, j));
        cs[j] = H(j, j) / r;
        sn[j] = H(j + 1, j) / r;
        H(j, j) = r;
        H(j + 1, j) = 0.0;
        g[j + 1] = -sn[j] * g[j];
        g[j] *= cs[j];
        ++j;
        if (!std::isfinite(g[j]))
            return 0;
        if (std::abs(g[j]) <= tol * bnorm)
        {
            const Vector y = H.topLeftCorner(j, j).triangularView<Eigen::Upper>().solve(g.head(j));
            b = Z.leftCols(j) * y;
            return j;
        }
    }
    return 0;
}

} // namespace

unsigned gmresHessenberg(const Matrix &M, Vector &b, double tol, unsigned maxIter, LinAlg::HessenbergWorkspace &hws,
                         Matrix &V, Matrix &Z)
{
    // P is the upper-Hessenberg part of M (LinAlg::hessenberg reads only the entries i <= j + 1)
    const auto n = M.rows();
    return gmres(
        n, [&M](const Vector &z) -> Vector { return M * z; },
        [&](Vector &z) { LinAlg::hessenberg(M.data(), z.data(), static_cast<uint32_t>(n), hws); }, b, tol, maxIter, V,
        Z);
}

void TridiagonalPlusSparse::setTridiagonal(double scale, const Vector &lo0, const Vector &d0, const Vector &up0)
{
    s = scale;
    lo = scale * lo0;
    d = scale * d0;
    up = scale * up0;
}

void TridiagonalPlusSparse::add(Eigen::Index i, Eigen::Index j, double v)
{
    if (j == i)
        d[i] += v;
    else if (j + 1 == i)
        lo[i] += v;
    else if (j == i + 1)
        up[i] += v;
    else
        throw std::logic_error("TridiagonalPlusSparse::add: entry off the tridiagonal.");
}

Vector TridiagonalPlusSparse::apply(const Vector &x) const
{
    Vector y = R * x;
    y *= s;
    const auto n = x.size();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        y[i] += d[i] * x[i];
        if (i > 0)
            y[i] += lo[i] * x[i - 1];
        if (i < n - 1)
            y[i] += up[i] * x[i + 1];
    }
    return y;
}

void TridiagonalPlusSparse::factor()
{
    const auto n = d.size();
    m_c.resize(n);
    m_inv.resize(n);
    m_inv[0] = 1.0 / d[0];
    m_c[0] = up[0] * m_inv[0];
    for (Eigen::Index i = 1; i < n; ++i)
    {
        m_inv[i] = 1.0 / (d[i] - lo[i] * m_c[i - 1]);
        m_c[i] = i < n - 1 ? up[i] * m_inv[i] : 0.0;
    }
}

void TridiagonalPlusSparse::solveTridiagonal(Vector &x) const
{
    const auto n = x.size();
    x[0] *= m_inv[0];
    for (Eigen::Index i = 1; i < n; ++i)
        x[i] = (x[i] - lo[i] * x[i - 1]) * m_inv[i];
    for (Eigen::Index i = n - 2; i >= 0; --i)
        x[i] -= m_c[i] * x[i + 1];
}

void TridiagonalPlusSparse::toDense(Matrix &M) const
{
    M = s * Matrix(R);
    const auto n = d.size();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        M(i, i) = d[i];
        if (i > 0)
            M(i, i - 1) = lo[i];
        if (i < n - 1)
            M(i, i + 1) = up[i];
    }
}

unsigned gmresTridiagonal(const TridiagonalPlusSparse &M, Vector &b, double tol, unsigned maxIter, Matrix &V, Matrix &Z)
{
    return gmres(
        b.size(), [&M](const Vector &z) { return M.apply(z); }, [&M](Vector &z) { M.solveTridiagonal(z); }, b, tol,
        maxIter, V, Z);
}

void TimeDependentBoltzmann::solveKrylov(Vector &b)
{
    if (m_luCurrent)
    {
        b = m_lu.solve(b);
        return;
    }
    const unsigned it = gmresHessenberg(m_work, b, m_opt.krylovTol, m_opt.krylovMaxIter, m_hws, m_krylovV, m_krylovZ);
    m_diag.krylovIterations += it ? it : m_opt.krylovMaxIter;
    if (it)
        return;
    if (m_diag.krylovFallbacks++ == 0)
        Log<Message>::Notify("timeDependent: GMRES did not reach ", m_opt.krylovTol, " in ", m_opt.krylovMaxIter,
                             " iterations; dense LU for this system (counted in krylovFallbacks).");
    m_lu.compute(m_work);
    m_luCurrent = true;
    b = m_lu.solve(b);
}

Vector TimeDependentBoltzmann::sparseA0Product(const Vector &f) const
{
    Vector r = m_stage.R * f;
    for (Grid::Index k = 0; k < m_n; ++k)
    {
        r[k] += m_A0d[k] * f[k];
        if (k > 0)
            r[k] += m_A0lo[k] * f[k - 1];
        if (k < m_n - 1)
            r[k] += m_A0up[k] * f[k + 1];
    }
    return r;
}

std::string TimeDependentBoltzmann::sparseFallbacksLine() const
{
    return m_solver == Solver::sparse ? "\nsparseFallbacks " + std::to_string(m_diag.sparseFallbacks) : std::string();
}

void TimeDependentBoltzmann::solveSparse(Vector &b)
{
    if (!m_denseCurrent)
    {
        const unsigned it = gmresTridiagonal(m_stage, b, m_opt.krylovTol, m_opt.krylovMaxIter, m_krylovV, m_krylovZ);
        m_diag.krylovIterations += it ? it : m_opt.krylovMaxIter;
        if (it)
            return;
        if (m_diag.sparseFallbacks++ == 0)
            Log<Message>::Notify("timeDependent: sparse GMRES did not reach ", m_opt.krylovTol, " in ",
                                 m_opt.krylovMaxIter, " iterations; dense solve for this system (counted in sparseFallbacks).");
        m_stage.toDense(m_work);
        m_denseCurrent = true;
        m_luCurrent = false;
    }
    if (m_denseSolver == Solver::hessenberg)
        LinAlg::hessenberg(m_work.data(), b.data(), m_n, m_hws);
    else
        solveKrylov(b);
}

void TimeDependentBoltzmann::stage(double a, double EoN2, const Vector &rhs, NonLinearState &nl, Vector &x,
                                   bool iterate)
{
    x = rhs;
    solveSystem(a, EoN2, nl, x);
    if (!iterate || (!m_nonConservative && !m_ee))
        return;
    // Picard iteration on the nonlinear coefficients
    for (unsigned it = 1;; ++it)
    {
        if (!x.allFinite())
            break;  // the step will be rejected
        const NonLinearState nlNew = nonLinearState(x);
        bool done = std::abs(nlNew.c - nl.c) <= m_opt.nonLinearTol * std::abs(nlNew.c) || nlNew.c == nl.c;
        nl = nlNew;
        const Vector xOld = x;
        if (done && !m_ee)
            break;
        x = rhs;
        solveSystem(a, EoN2, nl, x);
        if (m_ee)
        {
            const double fmax = x.cwiseAbs().maxCoeff();
            double dx = 0;
            for (Grid::Index i = 0; i < m_n; ++i)
                dx = std::max(dx, std::abs(x[i] - xOld[i]) / (std::abs(x[i]) + m_opt.atol * fmax));
            done = done && dx <= m_opt.eeTol;
            if (done)
                break;
        }
        if (it >= m_opt.maxNonLinearIter)
        {
            Log<Message>::Warning("timeDependent: nonlinear (CIEff/e-e) iteration did not converge at t = ", m_t, " s.");
            break;
        }
    }
}

bool TimeDependentBoltzmann::step(double h, double &errNorm)
{
    if (m_dynF1)
        return stepDynamic(h, errNorm);
    static const double g = 2.0 - std::sqrt(2.0);
    static const double d = (1.0 - g) / (2.0 - g);
    static const double k = (-3.0 * g * g + 4.0 * g - 2.0) / (12.0 * (2.0 - g));

    const NonLinearState nln = nonLinearState(m_f);
    evaluateField(nln.c);
    const Vector zn = applyNA(m_f, reducedField2(m_t), nln).cwiseQuotient(m_W);

    // nonlinear coefficients in the stages: iterated, lagged, or extrapolated from the last two accepted steps
    using NL = TimeDependentOptions::NonLinear;
    const bool extrapolate = m_opt.nonLinear == NL::extrapolate && m_tPrev >= 0 && m_t > m_tPrev;
    const bool iterate = m_opt.nonLinear == NL::iterate || (m_opt.nonLinear == NL::extrapolate && !extrapolate);
    auto guess = [&](double dt) {
        NonLinearState nl = nln;
        if (extrapolate)
        {
            const double w = dt / (m_t - m_tPrev);
            nl.c = nln.c + w * (nln.c - m_nlPrev.c);
            if (m_ee)
            {
                nl.A = nln.A + w * (nln.A - m_nlPrev.A);
                nl.B = nln.B + w * (nln.B - m_nlPrev.B);
            }
        }
        return nl;
    };

    // stage 1: trapezoidal rule over [t, t + g h]
    const Vector rhs1 = m_W.cwiseProduct(m_f + (0.5 * g * h) * zn);
    NonLinearState nlg = guess(g * h);
    Vector fg(m_n);
    stage(0.5 * g * h, reducedField2(m_t + g * h), rhs1, nlg, fg, iterate);
    const Vector zg = (2.0 / (g * h)) * (fg - m_f) - zn;

    // stage 2: BDF2 over [t, t + g h, t + h]
    const Vector rhs2 = m_W.cwiseProduct(fg / (g * (2.0 - g)) - ((1.0 - g) * (1.0 - g) / (g * (2.0 - g))) * m_f);
    NonLinearState nl1 = extrapolate ? guess(h) : nlg;
    Vector f1(m_n);
    stage(d * h, reducedField2(m_t + h), rhs2, nl1, f1, iterate);

    errNorm = 0.0;
    if (m_opt.fixedStep <= 0 && !m_frozen)
    {
        const Vector z1 = (f1 - rhs2.cwiseQuotient(m_W)) / (d * h);
        // local error estimate (Bank et al. 1985; Hosea and Shampine 1996), filtered by the stage-2 matrix
        Vector est = m_W.cwiseProduct((2.0 * k * h) * (zn / g - zg / (g * (1.0 - g)) + z1 / (1.0 - g)));
        resolve(est);
        if (!est.allFinite() || !f1.allFinite())
        {
            errNorm = std::numeric_limits<double>::infinity();  // e.g. a diverged nonlinear iteration: reject
            return false;
        }
        const double fmax = f1.cwiseAbs().maxCoeff();
        for (Grid::Index i = 0; i < m_n; ++i)
        {
            const double w = m_opt.rtol * std::max(std::abs(f1[i]), std::abs(m_f[i])) + m_opt.atol * fmax;
            errNorm = std::max(errNorm, std::abs(est[i]) / w);
        }
        if (!(errNorm <= 1.0))
            return false;
    }

    if (!f1.allFinite())
        throw std::runtime_error("timeDependent: non-finite EEDF with fixed steps (or a frozen schedule) at t = " +
                                 std::to_string(m_t) + " s; use smaller steps.");
    const double nrm = norm(f1);
    m_diag.maxNormDrift = std::max(m_diag.maxNormDrift, std::abs(nrm - 1.0));
    double pfOld = 0, pgOld = 0, e2Old = 0;
    Vector fOld;
    if (m_accumulate)
    {
        evaluateField(m_c);
        e2Old = reducedField2(m_t);
        pfOld = fieldPower(m_f, e2Old);
        pgOld = -m_c * meanEnergy(m_f);
        fOld = m_f;
    }
    m_cPrev = m_c;
    m_tPrev = m_t;
    m_nlPrev = nln;
    const double cOld = m_c;
    m_f = f1 / nrm;
    m_c = ciEff(m_f);
    m_t += h;
    if (m_opt.evolveElectronDensity && m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm)
        m_ne *= std::exp(0.5 * h * m_N * (cOld + m_c));
    if (m_accumulate)
        accumulate(h, fOld, pfOld, pgOld, e2Old);
    m_diag.lastStep = h;
    m_diag.minEedf = std::min(m_diag.minEedf, m_f.minCoeff());
    return true;
}

// ---------------------------------------------------------------- dynamic anisotropy

Vector TimeDependentBoltzmann::steadyStateHF(const Vector &f0, double ErmsSI, double WoN)
{
    // same fixed point as ElectronKineticsBoltzmann (temporal growth): A0 + Erms^2 F_HF(Omega_PT) + growth (+ e-e)
    Vector f = f0 / norm(f0);
    SparseMatrix F = m_F;
    double c = ciEff(f);
    for (unsigned it = 0; it < 500; ++it)
    {
        m_ek.fieldOperator.evaluate(m_ek.grid(), m_ek.mixture.collision_data().totalCrossSection(), 1.0, WoN,
                                    fieldGrowthTerm(c), F);
        Matrix M = m_A0;
        for (int k = 0; k < F.outerSize(); ++k)
            for (SparseMatrix::InnerIterator iF(F, k); iF; ++iF)
                M(iF.row(), iF.col()) += ErmsSI * ErmsSI * iF.value();
        const NonLinearState nl = nonLinearState(f);
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            if (m_nonConservative)
                M(k, k) -= (c / SI::gamma) * m_sqrtU[k];
            if (m_ee)
            {
                M(k, k) -= nl.A[k] + nl.B[k];
                if (k > 0)
                    M(k, k - 1) += nl.A[k - 1];
                if (k < m_n - 1)
                    M(k, k + 1) += nl.B[k + 1];
            }
        }
        Vector fNew(m_n);
        solveEEDF(fNew, M, m_ek.grid());
        const double cNew = ciEff(fNew);
        double df = 0;
        const double fmax = fNew.maxCoeff();
        for (Grid::Index i = 0; i < m_n; ++i)
            df = std::max(df, std::abs(fNew[i] - f[i]) / (std::abs(fNew[i]) + 1e-12 * fmax));
        f = m_ee ? (0.3 * fNew + 0.7 * f).eval() : fNew;
        f /= norm(f);
        const bool converged = df < 1e-9 && (!m_nonConservative || std::abs(cNew - c) <= 1e-10 * std::abs(cNew));
        c = 0.7 * cNew + 0.3 * c;
        if (converged)
            break;
    }
    m_Fc = std::numeric_limits<double>::quiet_NaN();  // m_F must be re-evaluated (WoN = 0)
    return f;
}

Vector TimeDependentBoltzmann::harmonicF1(const Vector &f, double A, double omega, double phase, double t,
                                          double c) const
{
    // (1/(N gamma sqrt u)) df1/dt + Omega f1 = -A sin(omega t + phase) D:
    // f1 = -A D Im[exp(i(omega t + phase)) / (Omega + i omega/(N gamma sqrt u))]
    const Vector g = fieldCoefficientNodes(c), D = gradientNodes(f);
    Vector f1 = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
    {
        const double om = m_uNode[j] / (3.0 * g[j]), w = omega / (m_N * SI::gamma * m_sqrtUNode[j]);
        const double th = omega * t + phase;
        f1[j] = -A * D[j] * (om * std::sin(th) - w * std::cos(th)) / (om * om + w * w);
    }
    return f1;
}

double TimeDependentBoltzmann::reducedFieldSI(double t) const
{
    return (*m_waveform)(t) * SI::Townsend;
}

Vector TimeDependentBoltzmann::fieldCoefficientNodes(double c) const
{
    // as FieldOperator::evaluate (WoN = 0): g_j = (u_j/3)/Omega_j, Omega_j = sigma_c,j + cEff/(gamma sqrt(u_j))
    const Vector &cs = m_ek.mixture.collision_data().totalCrossSection();
    const double cEff = fieldGrowthTerm(c);
    Vector g = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        g[j] = (1. / 3.) * m_uNode[j] / (cs[j] + cEff / (SI::gamma * m_sqrtUNode[j]));
    return g;
}

Vector TimeDependentBoltzmann::gradientNodes(const Vector &f) const
{
    Vector D = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        D[j] = (f[j] - f[j - 1]) / m_dn[j];
    return D;
}

Vector TimeDependentBoltzmann::divergence(const Vector &phi) const
{
    Vector r(m_n);
    for (Grid::Index k = 0; k < m_n; ++k)
        r[k] = (phi[k + 1] - phi[k]) / m_dc[k];
    return r;
}

Vector TimeDependentBoltzmann::quasiStaticF1(const Vector &f, double e, double c) const
{
    // f1 = -e D / Omega = -e D (3 g / u)
    const Vector g = fieldCoefficientNodes(c), D = gradientNodes(f);
    Vector f1 = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        f1[j] = -e * D[j] * 3.0 * g[j] / m_uNode[j];
    return f1;
}

Vector TimeDependentBoltzmann::f1Derivative(const Vector &f, const Vector &f1, double e, double c) const
{
    const Vector g = fieldCoefficientNodes(c), D = gradientNodes(f);
    Vector z = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
    {
        const double omega = m_uNode[j] / (3.0 * g[j]);
        z[j] = m_N * SI::gamma * m_sqrtUNode[j] * (-omega * f1[j] - e * D[j]);
    }
    return z;
}

Vector TimeDependentBoltzmann::applyNADynamic(const Vector &f, const Vector &f1, double e,
                                              const NonLinearState &nl) const
{
    Vector phi = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        phi[j] = -e * (m_uNode[j] / 3.0) * f1[j];
    Vector r = m_solver == Solver::sparse ? Vector(sparseA0Product(f) + divergence(phi))
                                          : Vector(m_A0 * f + divergence(phi));
    if (m_nonConservative)
        r -= (nl.c / SI::gamma) * m_sqrtU.cwiseProduct(f);
    if (m_ee)
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            r[k] -= (nl.A[k] + nl.B[k]) * f[k];
            if (k > 0)
                r[k] += nl.A[k - 1] * f[k - 1];
            if (k < m_n - 1)
                r[k] += nl.B[k + 1] * f[k + 1];
        }
    return m_N * r;
}

double TimeDependentBoltzmann::fieldPowerDynamic(const Vector &f1, double e) const
{
    Vector phi = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        phi[j] = -e * (m_uNode[j] / 3.0) * f1[j];
    return SI::gamma * m_uDu.dot(divergence(phi));
}

double TimeDependentBoltzmann::currentFieldPower() const
{
    return fieldPowerDynamic(m_f1, reducedFieldSI(m_t));
}

void TimeDependentBoltzmann::solveSystemDynamic(double a, double e, const Vector &r1, const NonLinearState &nl,
                                                const Vector &rhs0, Vector &x0, Vector &x1)
{
    // f1 = (r1 - a N gamma sqrt(u) e D(f0)) / (1 + a nu), nu = N gamma sqrt(u) Omega; substituted in the f0 stage:
    // W f0 - aN (A_rest f0 + div(geff D f0)) = rhs0 + aN div(phiHist), geff = e^2 g kappa, kappa = a nu/(1 + a nu)
    const Vector g = fieldCoefficientNodes(nl.c);
    Vector geff = Vector::Zero(m_n + 1), phiHist = Vector::Zero(m_n + 1), den = Vector::Ones(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
    {
        const double nu = m_N * SI::gamma * m_sqrtUNode[j] * m_uNode[j] / (3.0 * g[j]);
        den[j] = 1.0 + a * nu;
        geff[j] = e * e * g[j] * (a * nu / den[j]);
        phiHist[j] = -e * (m_uNode[j] / 3.0) * r1[j] / den[j];
    }
    const double s = -a * m_N;
    if (m_solver == Solver::sparse)
    {
        // the same terms as below, into the tridiagonal part of m_stage (see solveSystem)
        m_stage.setTridiagonal(s, m_A0lo, m_A0d, m_A0up);
        Vector &lo = m_stage.lo, &d = m_stage.d, &up = m_stage.up;
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            d[k] += m_W[k];
            if (m_nonConservative)
                d[k] += a * m_N * (nl.c / SI::gamma) * m_sqrtU[k];
            if (m_ee)
            {
                d[k] -= s * (nl.A[k] + nl.B[k]);
                if (k > 0)
                    lo[k] += s * nl.A[k - 1];
                if (k < m_n - 1)
                    up[k] += s * nl.B[k + 1];
            }
            if (k > 0)
            {
                const double w = geff[k] / (m_dn[k] * m_dc[k]);
                lo[k] += s * w;
                d[k] -= s * w;
            }
            if (k < m_n - 1)
            {
                const double w = geff[k + 1] / (m_dn[k + 1] * m_dc[k]);
                up[k] += s * w;
                d[k] -= s * w;
            }
        }
        m_stage.factor();
        m_denseCurrent = false;
    }
    else
    {
        m_work.noalias() = s * m_A0;
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            m_work(k, k) += m_W[k];
            if (m_nonConservative)
                m_work(k, k) += a * m_N * (nl.c / SI::gamma) * m_sqrtU[k];
            if (m_ee)
            {
                m_work(k, k) -= s * (nl.A[k] + nl.B[k]);
                if (k > 0)
                    m_work(k, k - 1) += s * nl.A[k - 1];
                if (k < m_n - 1)
                    m_work(k, k + 1) += s * nl.B[k + 1];
            }
            // field: (geff_{k+1} D_{k+1} - geff_k D_k)/dc_k, D_j = (f_j - f_{j-1})/dn_j
            if (k > 0)
            {
                const double w = geff[k] / (m_dn[k] * m_dc[k]);
                m_work(k, k - 1) += s * w;
                m_work(k, k) -= s * w;
            }
            if (k < m_n - 1)
            {
                const double w = geff[k + 1] / (m_dn[k + 1] * m_dc[k]);
                m_work(k, k + 1) += s * w;
                m_work(k, k) -= s * w;
            }
        }
    }
    x0 = rhs0 + (a * m_N) * divergence(phiHist);
    m_luCurrent = false;
    if (m_solver == Solver::lu)
        m_lu.compute(m_work);
    resolve(x0);
    const Vector D = gradientNodes(x0);
    x1 = Vector::Zero(m_n + 1);
    for (Grid::Index j = 1; j < m_n; ++j)
        x1[j] = (r1[j] - a * m_N * SI::gamma * m_sqrtUNode[j] * e * D[j]) / den[j];
}

void TimeDependentBoltzmann::stageDynamic(double a, double e, const Vector &rhs0, const Vector &r1,
                                          NonLinearState &nl, Vector &x0, Vector &x1, bool iterate)
{
    solveSystemDynamic(a, e, r1, nl, rhs0, x0, x1);
    if (!iterate || (!m_nonConservative && !m_ee))
        return;
    for (unsigned it = 1;; ++it)
    {
        if (!x0.allFinite())
            break;
        const NonLinearState nlNew = nonLinearState(x0);
        bool done = std::abs(nlNew.c - nl.c) <= m_opt.nonLinearTol * std::abs(nlNew.c) || nlNew.c == nl.c;
        nl = nlNew;
        const Vector xOld = x0;
        if (done && !m_ee)
            break;
        solveSystemDynamic(a, e, r1, nl, rhs0, x0, x1);
        if (m_ee)
        {
            const double fmax = x0.cwiseAbs().maxCoeff();
            double dx = 0;
            for (Grid::Index i = 0; i < m_n; ++i)
                dx = std::max(dx, std::abs(x0[i] - xOld[i]) / (std::abs(x0[i]) + m_opt.atol * fmax));
            if (done && dx <= m_opt.eeTol)
                break;
        }
        if (it >= m_opt.maxNonLinearIter)
        {
            Log<Message>::Warning("timeDependent: nonlinear (CIEff/e-e) iteration did not converge at t = ", m_t, " s.");
            break;
        }
    }
}

bool TimeDependentBoltzmann::stepDynamic(double h, double &errNorm)
{
    static const double g = 2.0 - std::sqrt(2.0);
    static const double d = (1.0 - g) / (2.0 - g);
    static const double k = (-3.0 * g * g + 4.0 * g - 2.0) / (12.0 * (2.0 - g));
    const double cg2 = 1.0 / (g * (2.0 - g)), cn2 = (1.0 - g) * (1.0 - g) / (g * (2.0 - g));

    const NonLinearState nln = nonLinearState(m_f);
    const double en = reducedFieldSI(m_t), eg = reducedFieldSI(m_t + g * h), e1 = reducedFieldSI(m_t + h);
    const Vector zn = applyNADynamic(m_f, m_f1, en, nln).cwiseQuotient(m_W);
    const Vector zn1 = f1Derivative(m_f, m_f1, en, nln.c);

    using NL = TimeDependentOptions::NonLinear;
    const bool extrapolate = m_opt.nonLinear == NL::extrapolate && m_tPrev >= 0 && m_t > m_tPrev;
    const bool iterate = m_opt.nonLinear == NL::iterate || (m_opt.nonLinear == NL::extrapolate && !extrapolate);
    auto guess = [&](double dt) {
        NonLinearState nl = nln;
        if (extrapolate)
        {
            const double w = dt / (m_t - m_tPrev);
            nl.c = nln.c + w * (nln.c - m_nlPrev.c);
            if (m_ee)
            {
                nl.A = nln.A + w * (nln.A - m_nlPrev.A);
                nl.B = nln.B + w * (nln.B - m_nlPrev.B);
            }
        }
        return nl;
    };

    // stage 1: trapezoidal rule over [t, t + g h]
    const double a1 = 0.5 * g * h;
    const Vector rhs1 = m_W.cwiseProduct(m_f + a1 * zn);
    const Vector r1g = m_f1 + a1 * zn1;
    NonLinearState nlg = guess(g * h);
    Vector fg(m_n), f1g;
    stageDynamic(a1, eg, rhs1, r1g, nlg, fg, f1g, iterate);
    const Vector zg = (2.0 / (g * h)) * (fg - m_f) - zn;
    const Vector zg1 = (2.0 / (g * h)) * (f1g - m_f1) - zn1;

    // stage 2: BDF2 over [t, t + g h, t + h]
    const Vector rhs2 = m_W.cwiseProduct(cg2 * fg - cn2 * m_f);
    const Vector r12 = cg2 * f1g - cn2 * m_f1;
    NonLinearState nl1 = extrapolate ? guess(h) : nlg;
    Vector f1v(m_n), f11;
    stageDynamic(d * h, e1, rhs2, r12, nl1, f1v, f11, iterate);

    errNorm = 0.0;
    if (m_opt.fixedStep <= 0 && !m_frozen)
    {
        const Vector z1 = (f1v - rhs2.cwiseQuotient(m_W)) / (d * h);
        const Vector z11 = (f11 - r12) / (d * h);
        Vector est = m_W.cwiseProduct((2.0 * k * h) * (zn / g - zg / (g * (1.0 - g)) + z1 / (1.0 - g)));
        resolve(est);
        Vector est1 = (2.0 * k * h) * (zn1 / g - zg1 / (g * (1.0 - g)) + z11 / (1.0 - g));
        if (!est.allFinite() || !f1v.allFinite() || !est1.allFinite() || !f11.allFinite())
        {
            errNorm = std::numeric_limits<double>::infinity();
            return false;
        }
        const double fmax = f1v.cwiseAbs().maxCoeff();
        for (Grid::Index i = 0; i < m_n; ++i)
        {
            const double w = m_opt.rtol * std::max(std::abs(f1v[i]), std::abs(m_f[i])) + m_opt.atol * fmax;
            errNorm = std::max(errNorm, std::abs(est[i]) / w);
        }
        // f1: filtered by its own implicit factor; weights from the f1 magnitude at the peak field
        const Vector gq = fieldCoefficientNodes(nl1.c);
        const double f1max = m_f1Scale.maxCoeff();
        for (Grid::Index j = 1; j < m_n; ++j)
        {
            const double nu = m_N * SI::gamma * m_sqrtUNode[j] * m_uNode[j] / (3.0 * gq[j]);
            const double w = m_opt.rtol * std::max({std::abs(f11[j]), std::abs(m_f1[j]), m_f1Scale[j]}) +
                             m_opt.atol * f1max;
            errNorm = std::max(errNorm, std::abs(est1[j] / (1.0 + d * h * nu)) / w);
        }
        if (!(errNorm <= 1.0))
            return false;
    }
    if (!f1v.allFinite() || !f11.allFinite())
        throw std::runtime_error("timeDependent: non-finite EEDF with fixed steps (or a frozen schedule) at t = " +
                                 std::to_string(m_t) + " s; use smaller steps.");

    const double nrm = norm(f1v);
    m_diag.maxNormDrift = std::max(m_diag.maxNormDrift, std::abs(nrm - 1.0));
    double pfOld = 0, pgOld = 0, e2Old = 0;
    Vector fOld;
    if (m_accumulate)
    {
        e2Old = en * en;
        pfOld = currentFieldPower();
        pgOld = -m_c * meanEnergy(m_f);
        fOld = m_f;
    }
    m_cPrev = m_c;
    m_tPrev = m_t;
    m_nlPrev = nln;
    const double cOld = m_c;
    m_f = f1v / nrm;
    m_f1 = f11 / nrm;
    m_c = ciEff(m_f);
    m_t += h;
    if (m_opt.evolveElectronDensity && m_opt.electronDensity == TimeDependentOptions::ElectronDensity::swarm)
        m_ne *= std::exp(0.5 * h * m_N * (cOld + m_c));
    if (m_accumulate)
        accumulate(h, fOld, pfOld, pgOld, e2Old);
    m_diag.lastStep = h;
    m_diag.minEedf = std::min(m_diag.minEedf, m_f.minCoeff());
    return true;
}

void TimeDependentBoltzmann::initialize()
{
    initializeImpl(nullptr, nullptr);
}

void TimeDependentBoltzmann::initialize(const Vector &u, const Vector &f0)
{
    if (u.size() != f0.size() || u.size() < 2)
        throw std::runtime_error("timeDependent: initialize(u, f0) needs energies and EEDF values of equal length >= 2.");
    initializeImpl(&u, &f0);
}

void TimeDependentBoltzmann::initializeImpl(const Vector *u0, const Vector *f00)
{
    const bool warm = u0 != nullptr;
    const Grid &grid = m_ek.grid();
    if (m_opt.gridFromPeakField && grid.smartGrid())
    {
        const double peak = m_waveform->peak(0.0, horizon());
        Log<Message>::Notify("timeDependent: choosing the energy grid with the smart grid at the peak field ", peak,
                             " Td.");
        m_wc.updateReducedField(peak);
        m_ek.solveSmartGrid();
    }
    assembleFixedOperators();

    if (warm)
    {
        // log-linear interpolation of the given f0 onto the cell energies; 0 above its last energy
        const Vector &u = *u0, &f = *f00;
        const Vector &cells = grid.getCells();
        m_f.resize(m_n);
        Grid::Index j = 0;
        for (Grid::Index k = 0; k < m_n; ++k)
        {
            const double e = cells[k];
            if (e > u[u.size() - 1])
            {
                m_f[k] = 0.0;
                continue;
            }
            while (j + 2 < u.size() && u[j + 1] < e)
                ++j;
            const double w = std::clamp((e - u[j]) / (u[j + 1] - u[j]), 0.0, 1.0);
            const double a = std::log(std::max(f[j], 1e-300)), b = std::log(std::max(f[j + 1], 1e-300));
            m_f[k] = std::exp(a + w * (b - a));
        }
        Log<Message>::Notify("timeDependent: initial state is the given EEDF (warm start).");
    }
    else if (m_opt.initial == TimeDependentOptions::Initial::steadyState)
    {
        double e0 = (*m_waveform)(0.0);
        if (m_opt.initialReducedField == "rms")
            e0 = m_waveform->rms(0.0, horizon());
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
    m_ne = m_wc.electronDensity();
    double sa = 0, sw = 0, sphi = 0, soff = 0;
    const bool sinusoid = m_waveform->sinusoid(sa, sw, sphi, soff) && soff == 0.0;
    const bool harmonicStart = m_dynF1 && sinusoid &&
                               (warm || (m_opt.initial == TimeDependentOptions::Initial::steadyState &&
                                         m_opt.initialReducedField == "rms"));
    if (harmonicStart && !warm)
    {
        // LoKI-B's HF steady state at E_rms and omega, and the harmonic f1: the periodic state when omega >> nu_u
        Log<Message>::Notify("timeDependent: initial state is the HF steady solution at E_rms = ",
                             std::abs(sa) / std::sqrt(2.0), " Td, f = ", sw / (2 * Constant::pi), " Hz.");
        m_f = steadyStateHF(m_f, std::abs(sa) / std::sqrt(2.0) * SI::Townsend, sw / m_N);
        m_c = ciEff(m_f);
    }
    if (m_dynF1)
    {
        m_f1 = harmonicStart ? harmonicF1(m_f, sa * SI::Townsend, sw, sphi, 0.0, m_c)
                             : quasiStaticF1(m_f, reducedFieldSI(0.0), m_c);
        const double emax = m_waveform->peak(0.0, horizon()) * SI::Townsend;
        m_f1Scale = quasiStaticF1(m_f, emax, m_c).cwiseAbs();
    }
    m_t = 0.0;
    m_tPrev = -1.0;
    m_nextSample = 0;
    // periodic-mode state (a repeated initialize() starts a new periodic run)
    m_frozen = false;
    m_schedule.clear();
    m_hPeriodStart = 0.0;
    m_diag = Diagnostics();
    m_diag.minEedf = m_f.minCoeff();

    if (m_opt.fixedStep > 0)
        m_h = m_opt.fixedStep;
    else if (m_opt.initialStep > 0)
        m_h = m_opt.initialStep;
    else
    {
        const NonLinearState nl = nonLinearState(m_f);
        evaluateField(nl.c);
        const Vector z = applyNA(m_f, reducedField2(0.0), nl).cwiseQuotient(m_W);
        const double fmax = m_f.cwiseAbs().maxCoeff();
        double r = 0;
        for (Grid::Index i = 0; i < m_n; ++i)
            r = std::max(r, std::abs(z[i]) / (std::abs(m_f[i]) + m_opt.atol * fmax));
        m_h = r > 0 ? 1e-2 * std::cbrt(m_opt.rtol) / r : 1e-3 * horizon();
        m_h = std::min(m_h, 0.1 * horizon());
    }
}

void TimeDependentBoltzmann::advance(double tEnd)
{
    std::vector<double> stops = m_waveform->breakpoints(m_t, tEnd);
    for (double t : m_opt.sampleTimes)
        if (t > m_t && t < tEnd)
            stops.push_back(t);
    for (double t : m_extraStops)
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
        if (m_frozen || m_t + h >= target - eps)
            h = target - m_t;
        else if (m_t + 1.5 * h > target)
            h = 0.5 * (target - m_t);

        double err = 0;
        const bool ok = step(h, err);
        ++m_diag.steps;
        if (m_diag.steps > m_opt.maxSteps)
            throw std::runtime_error("timeDependent: maxSteps exceeded.");
        if (ok && m_recordSchedule)
            m_schedule.push_back(m_t - m_periodStart);
        if (ok && m_recordFirstStep)
        {
            m_hPeriodStart = h;
            m_recordFirstStep = false;
        }
        if (m_opt.fixedStep <= 0 && !m_frozen)
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
                    "Pbalance(eVm3/s) depsdt(eV/s) powerResidual minEedf steps rejected solves lastStep(s) ne(m-3)\n";
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
    prepareOutputState();
    m_ek.evaluatePower();
    if (m_dynF1)
        setFieldPower(m_ek.power, currentFieldPower());
    m_ek.mixture.collision_data().evaluateRateCoefficients(grid, m_ek.eedf);
    m_ek.evaluateSwarmParameters();

    const NonLinearState nl = nonLinearState(m_f);
    evaluateField(nl.c);
    const Vector z = (m_dynF1 ? applyNADynamic(m_f, m_f1, reducedFieldSI(m_t), nl)
                              : applyNA(m_f, reducedField2(m_t), nl)).cwiseQuotient(m_W);
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
                     << m_diag.lastStep << ' ' << m_ne << '\n';
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
    prepareOutputState();
    m_ek.evaluatePower();
    if (m_dynF1)
        setFieldPower(m_ek.power, currentFieldPower());
    m_ek.mixture.collision_data().evaluateRateCoefficients(grid, m_ek.eedf);
    m_ek.evaluateSwarmParameters();
    m_ek.evaluateFirstAnisotropy();
    if (m_dynF1)
        for (Grid::Index k = 0; k < m_n; ++k)
            m_ek.firstAnisotropy[k] = 0.5 * (m_f1[k] + m_f1[k + 1]);  // faces -> cells
    m_ek.obtainedNewEedf.emit(grid, m_ek.eedf, m_wc, m_ek.power, m_ek.mixture.collision_data(), m_ek.swarmParameters,
                              &m_ek.firstAnisotropy);
}

double TimeDependentBoltzmann::horizon() const
{
    return m_opt.periodic ? m_waveform->period() : m_opt.finalTime;
}

double TimeDependentBoltzmann::fieldPower(const Vector &f, double EoN2) const
{
    return SI::gamma * EoN2 * m_uDu.dot(m_F * f);
}

void TimeDependentBoltzmann::resetAccumulators()
{
    m_accF = Vector::Zero(m_n);
    m_accT = m_accC = m_accPfield = m_accPgrowth = m_accE2 = 0.0;
}

void TimeDependentBoltzmann::accumulate(double h, const Vector &fOld, double pfOld, double pgOld, double e2Old)
{
    evaluateField(m_c);
    const double e2 = reducedField2(m_t);
    m_accT += h;
    m_accF += (0.5 * h) * (fOld + m_f);
    m_accC += 0.5 * h * (ciEff(fOld) + m_c);
    m_accPfield += 0.5 * h * (pfOld + (m_dynF1 ? currentFieldPower() : fieldPower(m_f, e2)));
    m_accPgrowth += 0.5 * h * (pgOld - m_c * meanEnergy(m_f));
    m_accE2 += 0.5 * h * (e2Old + e2);
}

void TimeDependentBoltzmann::runPeriodic()
{
    const auto begin = std::chrono::steady_clock::now();
    const double T = m_waveform->period();
    std::ofstream conv;
    if (!m_outFolder.empty())
    {
        conv.open(m_outFolder / "periodConvergence.txt");
        conv << "# period t_end(s) delta ratio remainingEstimate CIEffbar(m3/s) meanEnergybar(eV) Pfieldbar(eVm3/s) "
                "steps\n" << std::scientific << std::setprecision(8);
    }
    Vector fbarOld;
    double cOld = 0, pOld = 0, deltaOld = 0;
    unsigned p = 0, compared = 0;  // compared: period-to-period comparisons on the current (adaptive or frozen) map
    bool converged = false;
    for (; p < m_opt.maxPeriods && !converged; ++p)
    {
        const double t0 = m_t;
        const bool frozenBefore = m_frozen;
        m_opt.sampleTimes = {t0 + T};  // one timeSeries row per period
        m_nextSample = 0;
        resetAccumulators();
        m_accumulate = true;
        advancePeriod(t0, T);
        m_accumulate = false;
        const Vector fbar = m_accF / m_accT;
        const double cbar = m_accC / m_accT, pbar = m_accPfield / m_accT;
        double delta = std::numeric_limits<double>::infinity(), ratio = 0, remaining = 0;
        if (p > 0)
        {
            const double fmax = fbar.cwiseAbs().maxCoeff();
            delta = 0;
            for (Grid::Index i = 0; i < m_n; ++i)
                delta = std::max(delta, std::abs(fbar[i] - fbarOld[i]) / (std::abs(fbar[i]) + m_opt.periodicAtol * fmax));
            if (m_nonConservative && cbar != 0)
                delta = std::max(delta, std::abs(cbar - cOld) / std::abs(cbar));
            delta = std::max(delta, std::abs(pbar - pOld) / std::abs(pbar));
            // the first period on the frozen schedule is compared with an adaptive one: not used for the estimate
            if (frozenBefore || !m_frozen)
                ++compared;
            ratio = compared >= 2 && deltaOld > 0 ? delta / deltaOld : 1.0;
            remaining = ratio < 1 ? delta * ratio / (1 - ratio) : 10 * delta;
            converged = p + 1 >= m_opt.minPeriods && compared >= 2 && delta < m_opt.periodicTol &&
                        remaining < m_opt.periodicTol;
            if (!converged && !m_frozen && m_opt.fixedStep <= 0 && delta < m_opt.freezeTol)
            {
                m_frozen = true;  // reuse this period's step schedule from now on
                compared = 0;
                Log<Message>::Notify("timeDependent: step schedule frozen after period ", p + 1, " (",
                                     m_schedule.size(), " steps per period).");
            }
        }
        if (conv.is_open())
            conv << p + 1 << ' ' << m_t << ' ' << delta << ' ' << ratio << ' ' << remaining << ' ' << cbar << ' '
                 << meanEnergy(fbar) << ' ' << pbar << ' ' << m_diag.steps << ' ' << (m_frozen ? 1 : 0) << '\n'
                 << std::flush;
        fbarOld = fbar;
        cOld = cbar;
        pOld = pbar;
        deltaOld = delta;
    }
    m_periodicResult = PeriodicResult();
    m_periodicResult.converged = converged;
    if (!converged)
        Log<Message>::Warning("timeDependent: no periodic steady state after ", p, " periods (periodic.tol = ",
                              m_opt.periodicTol, ").");
    else
        Log<Message>::Notify("timeDependent: periodic steady state after ", p, " periods.");

    // averaging window: averagePeriods periods, phase-resolved samples in the last one
    const double tAvg0 = m_t;
    resetAccumulators();
    m_accumulate = true;
    for (unsigned q = 0; q < m_opt.averagePeriods; ++q)
    {
        const double t0 = m_t;
        m_opt.sampleTimes.clear();
        if (q + 1 == m_opt.averagePeriods)
            for (unsigned j = 1; j <= m_opt.samplesPerPeriod; ++j)
                m_opt.sampleTimes.push_back(t0 + T * j / m_opt.samplesPerPeriod);
        else
            m_opt.sampleTimes.push_back(t0 + T);
        m_nextSample = 0;
        if (q + 1 == m_opt.averagePeriods)
        {
            // the phase-resolved EEDFs are those of the last period only
            m_eedfTimes.clear();
            m_eedfFields.clear();
            m_eedfSamples.clear();
            sample();
        }
        advancePeriod(t0, T);
    }
    m_accumulate = false;
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    writeCycleAverage(p + m_opt.averagePeriods, tAvg0, wall);
}

void TimeDependentBoltzmann::advancePeriod(double t0, double T)
{
    m_extraStops.clear();
    m_periodStart = t0;
    if (m_frozen)
    {
        for (double s : m_schedule)
            if (s < T * (1 - 1e-12))
                m_extraStops.push_back(t0 + s);
    }
    else
    {
        for (unsigned j = 1; j < m_opt.stepsPerPeriod; ++j)
            m_extraStops.push_back(t0 + T * j / m_opt.stepsPerPeriod);
        if (m_opt.fixedStep <= 0)
        {
            // the same first step in every period (a deterministic step sequence for a periodic state): the first
            // accepted step of the first period, which starts from T/stepsPerPeriod
            if (m_hPeriodStart <= 0)
            {
                m_h = T / m_opt.stepsPerPeriod;
                m_recordFirstStep = true;
            }
            else
                m_h = m_hPeriodStart;
        }
        m_schedule.clear();
        m_recordSchedule = m_opt.fixedStep <= 0;
    }
    advance(t0 + T);
    m_recordSchedule = false;
    m_extraStops.clear();
}

void TimeDependentBoltzmann::writeCycleAverage(unsigned periods, double t0, double wall)
{
    const Grid &grid = m_ek.grid();
    const Vector fbar = m_accF / m_accT;
    const double cbar = m_accC / m_accT, pfield = m_accPfield / m_accT, pgrowth = m_accPgrowth / m_accT;
    const double erms = m_waveform->rms(0.0, m_waveform->period());  // exact; m_accE2 is the quadrature value
    const double ermsQuad = std::sqrt(m_accE2 / m_accT) / SI::Townsend;

    // LoKI-B files for the cycle-averaged state: the EEDF, rate coefficients and every power term that is linear
    // in f come from fbar; the field and growth powers (nonlinear in time) are the time averages.
    m_wc.setCurrentJobFolder("td_cycleAverage");
    m_wc.updateReducedField(erms);
    m_ek.eedf = fbar;
    m_ek.CIEff = cbar;
    prepareOutputState();
    m_ek.evaluatePower();
    Power &pw = m_ek.power;
    pw.field = pfield;
    if (m_nonConservative)
        pw.eDensGrowth = pgrowth;
    pw.balance = pw.field + pw.elasticNet + pw.carNet + pw.inelastic + pw.superelastic + pw.eDensGrowth +
                 pw.electronElectronNet;
    pw.reference = pw.field + std::max(pw.elasticGain, 0.0) + std::max(pw.carGain, 0.0) + std::max(pw.superelastic, 0.0);
    pw.relativeBalance = std::abs(pw.balance) / pw.reference;
    m_ek.mixture.collision_data().evaluateRateCoefficients(grid, m_ek.eedf);
    m_ek.evaluateSwarmParameters();
    m_ek.evaluateFirstAnisotropy();
    m_ek.obtainedNewEedf.emit(grid, m_ek.eedf, m_wc, m_ek.power, m_ek.mixture.collision_data(), m_ek.swarmParameters,
                              &m_ek.firstAnisotropy);

    std::ostringstream os;
    os << std::scientific << std::setprecision(10);
    os << "periods " << periods << "\naveragingWindow " << t0 << ' ' << m_t << "\nperiod " << m_waveform->period()
       << "\nreducedFieldRMS(Td) " << erms << "\nreducedFieldRMSQuadrature(Td) " << ermsQuad << "\nmeanEnergy(eV) " << meanEnergy(fbar) << "\nCIEff(m3/s) " << cbar
       << "\nPfield(eVm3/s) " << pfield << "\nPeDensGrowth(eVm3/s) " << pgrowth << "\nPelastic(eVm3/s) " << pw.elasticNet
       << "\nPinelastic(eVm3/s) " << pw.inelastic << "\nPsuperelastic(eVm3/s) " << pw.superelastic
       << "\nPbalance(eVm3/s) " << pw.balance << "\nrelativeBalance " << pw.relativeBalance << "\nsteps "
       << m_diag.steps << "\nrejected " << m_diag.rejected << "\nsolves " << m_diag.solves << "\nkrylovIterations " << m_diag.krylovIterations
       << "\nkrylovFallbacks " << m_diag.krylovFallbacks << sparseFallbacksLine() << "\nmaxNormDrift "
       << m_diag.maxNormDrift << "\ncells " << m_n << "\nuMax " << grid.uMax() << "\nwallTime " << wall << " s\n";
    Log<Message>::Notify("timeDependent cycle average:\n", os.str());
    PeriodicResult &res = m_periodicResult;
    res.periods = periods;
    res.reducedFieldRMS = erms;
    res.energy = grid.getCells();
    res.eedf = fbar;
    res.meanEnergy = meanEnergy(fbar);
    res.CIEff = cbar;
    res.power = pw;
    res.swarm = m_ek.swarmParameters;
    res.rates = m_ek.mixture.collision_data().rateCoefficients();
    res.eedfEnd = m_f;
    res.wallTime = wall;
    if (!m_outFolder.empty())
        std::ofstream(m_outFolder / "cycleAverage.txt") << os.str();
}

void TimeDependentBoltzmann::prepareOutputState()
{
    // the field operator (g) and e-e coefficients that LoKI-B's power evaluation uses, for m_ek.eedf and m_ek.CIEff
    m_ek.fieldOperator.evaluate(m_ek.grid(), m_ek.mixture.collision_data().totalCrossSection(), 1.0, 0.0,
                                fieldGrowthTerm(m_ek.CIEff));
    if (m_ee)
        m_ek.eeOperator->update_g_ee_AB(m_ek.grid(), m_ek.eedf, m_ne, m_N);
}

void TimeDependentBoltzmann::setPopulations(const std::map<std::string, double> &populations)
{
    auto &composition = m_ek.mixture.composition();
    for (const auto &[id, pop] : populations)
    {
        Gas::State *state = composition.findStateById(id);
        if (!state)
            throw std::runtime_error("timeDependent::setPopulations: unknown state '" + id + "'.");
        state->setPopulation(pop);
    }
    composition.checkPopulations();
    composition.evaluateReducedDensities();
    m_ek.evaluateMatrix();
    assembleFixedOperators();
    m_c = ciEff(m_f);
    m_tPrev = -1.0;  // no extrapolation across the change
}

void TimeDependentBoltzmann::setElectronDensity(double ne, double dlnneDt)
{
    m_ne = ne;
    m_dlnneDt = dlnneDt;
    m_tPrev = -1.0;
}

std::vector<RateCoefficient> TimeDependentBoltzmann::rateCoefficients()
{
    m_ek.mixture.collision_data().evaluateRateCoefficients(m_ek.grid(), m_f);
    return m_ek.mixture.collision_data().rateCoefficients();
}

void TimeDependentBoltzmann::run()
{
    const auto begin = std::chrono::steady_clock::now();
    initialize();
    openOutput();
    sample();
    if (!m_opt.periodic)
        ++m_nextSample;  // sampleTimes[0] == 0
    emitState("td_initial");
    if (m_opt.periodic)
        runPeriodic();
    else
        advance(m_opt.finalTime);
    emitState("td_final");
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::ostringstream summary;
    summary << "finalTime " << m_t << " s\nsteps " << m_diag.steps << "\nrejected " << m_diag.rejected
            << "\nsolves " << m_diag.solves << "\nkrylovIterations " << m_diag.krylovIterations
       << "\nkrylovFallbacks " << m_diag.krylovFallbacks << sparseFallbacksLine() << "\nmaxNormDrift " << m_diag.maxNormDrift << "\nminEedf "
            << m_diag.minEedf << "\ncells " << m_n << "\nuMax " << m_ek.grid().uMax() << "\nwallTime " << elapsed
            << " s\n";
    Log<Message>::Notify("timeDependent finished:\n", summary.str());
    if (m_outFolder.empty())
        return;
    std::ofstream(m_outFolder / "summary.txt") << summary.str();
    writeEedfSamples();
}

void TimeDependentBoltzmann::writeEedfSamples()
{
    if (m_outFolder.empty())
        return;
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
