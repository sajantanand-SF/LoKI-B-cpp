/** \file
 *
 *  Time-dependent solution of the two-term electron Boltzmann equation for the
 *  isotropic EEDF f(u,t) under a prescribed reduced field E(t)/N.
 *
 *  The anisotropy f1 is quasi-static (slaved to f and E(t)), as in the MATLAB
 *  LoKI-B time-dependent mode (Tejero-del-Caz et al., PSST 30 065008 (2021),
 *  eqs. 3a,3b). The semi-discrete system is
 *
 *      W df/dt = N A(t,f) f,   W = diag(sqrt(u_k)/gamma),
 *      A = A0 + (E/N)^2 F[Omega_c] + D[<nu_eff>/N],
 *
 *  where A0 collects the elastic, CAR, inelastic/superelastic, ionization and
 *  attachment matrices of ElectronKineticsBoltzmann, F is its field matrix at
 *  E/N = 1, and D = diag(-(CIEff/gamma) sqrt(u_k)) is its temporal-growth
 *  diagonal. It is integrated with TR-BDF2 (Hosea and Shampine 1996) with
 *  adaptive steps. The energy grid is fixed during a run.
 *
 *  This mode is used only when electronKinetics.timeDependent.isOn is true;
 *  otherwise none of this code is reached. See docs/design.md in the TDBE
 *  simulation folder for the design and validation plan.
 */

#ifndef LOKI_CPP_TIMEDEPENDENT_H
#define LOKI_CPP_TIMEDEPENDENT_H

#include "LoKI-B/ElectronKinetics.h"
#include "LoKI-B/LinearAlgebra.h"
#include "LoKI-B/json.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace loki
{

/// A reduced field E/N (in Td) as a function of time t (in s).
class Waveform
{
public:
    virtual ~Waveform() = default;
    virtual double operator()(double t) const = 0;
    /// times inside (t0,t1) where the waveform has a kink; steps land on them
    virtual std::vector<double> breakpoints(double t0, double t1) const { return {}; }
    /// the period in s, or 0 for a non-periodic waveform
    virtual double period() const { return 0.0; }
    /// max |E/N| over [t0,t1]
    virtual double peak(double t0, double t1) const;
    /// RMS of E/N over one period (periodic) or over [t0,t1]
    virtual double rms(double t0, double t1) const;
    static std::unique_ptr<Waveform> create(const json_type &cnf, const std::filesystem::path &basePath);
};

struct TimeDependentOptions
{
    explicit TimeDependentOptions(const json_type &cnf);

    enum class ElectronDensity { swarm, constant };
    ElectronDensity electronDensity = ElectronDensity::swarm;

    enum class Initial { steadyState, maxwellian };
    Initial initial = Initial::steadyState;
    /// steadyState: "initial" (E(0)), "rms", or a value in Td (initialReducedFieldValue)
    std::string initialReducedField = "initial";
    double initialReducedFieldValue = 0.0;
    /// maxwellian: temperature in eV (<=0: the gas temperature)
    double initialTemperature = -1.0;

    bool gridFromPeakField = true;

    double finalTime = 0.0;
    std::vector<double> sampleTimes;

    double rtol = 1e-4;
    double atol = 1e-12;
    double maxStep = 0.0;      // 0: no limit
    double initialStep = 0.0;  // 0: automatic
    double fixedStep = 0.0;    // > 0: fixed steps, no error control
    bool lagNonLinear = false; // true: CIEff from the start of the step
    unsigned maxNonLinearIter = 20;
    double nonLinearTol = 1e-10;
    unsigned long maxSteps = 100000000;
};

class TimeDependentBoltzmann
{
public:
    TimeDependentBoltzmann(ElectronKineticsBoltzmann &ek, WorkingConditions &wc, const json_type &cnf,
                           const std::filesystem::path &basePath);

    /** Run the whole time-dependent simulation described in the input: choose
     *  the grid, build the initial state, integrate to finalTime, write the
     *  sampled results to outputFolder, and emit obtainedNewEedf for the
     *  initial and final states (job folders 'td_initial', 'td_final') so the
     *  usual LoKI-B output files are written for them.
     */
    void run();

    // stepper interface (also used by run())
    void initialize();
    /// advance the state to tEnd under the configured waveform
    void advance(double tEnd);

    double time() const { return m_t; }
    const Vector &eedf() const { return m_f; }
    double CIEff() const { return m_c; }

private:
    struct Diagnostics
    {
        unsigned long steps = 0, rejected = 0, solves = 0;
        double maxNormDrift = 0.0;
        double lastStep = 0.0;
        double minEedf = 0.0;
    };

    void checkSupported() const;
    void assembleFixedOperators();
    void evaluateField(double c);
    double reducedField2(double t) const;
    double ciEff(const Vector &f) const;
    double norm(const Vector &f) const;
    /// N*A(t,c)*f
    Vector applyNA(const Vector &f, double EoN2, double c) const;
    /// solve (W - a*N*A) x = rhs; the system matrix is kept in m_work
    void solveSystem(double a, double EoN2, double c, Vector &x);
    /// re-solve with the matrix currently in m_work
    void resolve(Vector &x);
    /// implicit stage: (W - a N A(c)) x = rhs, with c = CIEff(x) iterated unless lagged
    void stage(double a, double EoN2, const Vector &rhs, double &c, Vector &x);
    bool step(double h, double &errNorm);
    void sample();
    void openOutput();
    void emitState(const std::string &jobFolder);

    ElectronKineticsBoltzmann &m_ek;
    WorkingConditions &m_wc;
    TimeDependentOptions m_opt;
    std::unique_ptr<Waveform> m_waveform;
    const double m_N;
    Grid::Index m_n;

    Vector m_W;        // sqrt(u_k)/gamma
    Vector m_sqrtU;
    Vector m_numberWeights;  // sqrt(u_k) du_k, for the normalization
    Matrix m_A0;
    SparseMatrix m_F;  // field matrix at E/N = 1 for the current CIEff (Omega_c)
    double m_Fc = -1.0;  // CIEff used for m_F
    Vector m_coefsCI;
    bool m_nonConservative = false;
    enum class Solver { tdma, hessenberg, lu } m_solver = Solver::lu;
    Matrix m_work;
    LinAlg::HessenbergWorkspace m_hws;
    Eigen::PartialPivLU<Matrix> m_lu;

    Vector m_f;
    double m_t = 0.0;
    double m_c = 0.0;
    double m_h = 0.0;
    std::size_t m_nextSample = 0;
    Diagnostics m_diag;

    std::filesystem::path m_outFolder;
    std::ofstream m_timeSeries, m_rates;
    std::vector<double> m_eedfTimes, m_eedfFields;
    std::vector<Vector> m_eedfSamples;
};

} // namespace loki

#endif // LOKI_CPP_TIMEDEPENDENT_H
