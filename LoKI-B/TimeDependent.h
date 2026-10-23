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
#include <map>
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

    /** The f1 growth term Omega_c = sigma_c + (1/(gamma sqrt(u))) (1/N) dln(ne)/dt (the f equation always has
     *  the <nu_eff> f term): swarm uses dln(ne)/dt = <nu_eff> (as steady temporal growth), constant uses 0,
     *  external uses the value passed with setElectronDensity() (coupling to a chemistry model).
     */
    enum class ElectronDensity { swarm, constant, external };
    ElectronDensity electronDensity = ElectronDensity::swarm;
    /// swarm: integrate dne/dt = <nu_eff> ne (used by e-e collisions); otherwise ne stays at its input value
    bool evolveElectronDensity = false;

    enum class Initial { steadyState, maxwellian };
    Initial initial = Initial::steadyState;
    /// steadyState: "initial" (E(0)), "rms", or a value in Td (initialReducedFieldValue)
    std::string initialReducedField = "initial";
    double initialReducedFieldValue = 0.0;
    /// maxwellian: temperature in eV (<=0: the gas temperature)
    double initialTemperature = -1.0;

    bool gridFromPeakField = true;

    double finalTime = 0.0;  // required unless periodic
    std::vector<double> sampleTimes;

    /** Periodic steady state (needs a periodic waveform): march period by period until the period-averaged
     *  state stops changing, then average over averagePeriods more periods and write phase-resolved samples
     *  of the last one. Convergence: delta_p = max over cells of |fbar_p - fbar_{p-1}| / (|fbar_p| +
     *  periodicAtol max fbar_p), also over <nu_eff>/N and the field power; with r = delta_p/delta_{p-1}, the
     *  remaining change is estimated as delta_p r/(1-r) (10 delta_p if r >= 1), and the run is converged when
     *  both are below periodicTol, after at least minPeriods.
     */
    bool periodic = false;
    double periodicTol = 1e-5;
    double periodicAtol = 1e-10;
    unsigned minPeriods = 3;
    unsigned maxPeriods = 100000;
    unsigned averagePeriods = 1;
    unsigned samplesPerPeriod = 32;
    /// steps land on stepsPerPeriod equidistant phase points (also the default maxStep), and each period starts
    /// with this step, so that a periodic state gives the same step sequence (and period averages) every period
    unsigned stepsPerPeriod = 32;
    /** Once delta_p < freezeTol, the accepted step times of that period (relative to its start) are reused in every
     *  later period without error control. The period map is then smooth, with a true fixed point; with adaptive
     *  steps, accept/reject decisions near err = 1 can otherwise make it alternate between two step sequences
     *  (a numerical period-2 cycle whose amplitude sets a floor on delta_p).
     */
    double freezeTol = 1e-3;

    double rtol = 1e-4;
    double atol = 1e-12;
    double maxStep = 0.0;      // 0: no limit
    double initialStep = 0.0;  // 0: automatic
    double fixedStep = 0.0;    // > 0: fixed steps, no error control
    /** Nonlinear coefficients (CIEff, and the e-e coefficients A, B) in the implicit stages: iterate (Picard, to
     *  nonLinearTol on CIEff and eeTol on f), lag (from the start of the step), or extrapolate linearly in time from
     *  the last two steps (one solve per stage).
     */
    enum class NonLinear { iterate, lag, extrapolate };
    NonLinear nonLinear = NonLinear::iterate;
    unsigned maxNonLinearIter = 50;
    double nonLinearTol = 1e-10;
    double eeTol = 1e-9;
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
    double electronDensity() const { return m_ne; }
    double meanEnergy() const { return meanEnergy(m_f); }

    /** Coupling hooks, to be called between advance() calls (e.g. by a chemistry model).
     *  setPopulations: new populations of states, by state id (e.g. {"He(1S1)", 0.999}, {"He(2S3)", 1e-3}); the
     *  populations of each gas must still add up to 1. The collision operators are rebuilt (cost: one
     *  steady-state matrix evaluation) and the current EEDF is kept as the state.
     *  setElectronDensity: ne (used by e-e collisions) and, for electronDensity external, dln(ne)/dt (1/s).
     */
    void setPopulations(const std::map<std::string, double> &populations);
    void setElectronDensity(double ne, double dlnneDt = 0.0);
    /// rate coefficients (inelastic, superelastic) of the current EEDF, in the order of the LoKI-B output
    std::vector<RateCoefficient> rateCoefficients();

private:
    struct Diagnostics
    {
        unsigned long steps = 0, rejected = 0, solves = 0;
        double maxNormDrift = 0.0;
        double lastStep = 0.0;
        double minEedf = 0.0;
    };

    /// the coefficients of A that depend on f: CIEff and the e-e coefficients A, B (empty without e-e)
    struct NonLinearState
    {
        double c = 0.0;
        Vector A, B;
    };
    void checkSupported() const;
    void assembleFixedOperators();
    void evaluateField(double c);
    double fieldGrowthTerm(double c) const;
    double reducedField2(double t) const;
    double ciEff(const Vector &f) const;
    double norm(const Vector &f) const;
    /// CIEff and e-e coefficients of the EEDF f (normalized inside)
    NonLinearState nonLinearState(const Vector &f) const;
    /// N*A(t,nl)*f
    Vector applyNA(const Vector &f, double EoN2, const NonLinearState &nl) const;
    /// solve (W - a*N*A) x = rhs; the system matrix is kept in m_work
    void solveSystem(double a, double EoN2, const NonLinearState &nl, Vector &x);
    /// re-solve with the matrix currently in m_work
    void resolve(Vector &x);
    /// implicit stage: (W - a N A(nl)) x = rhs, with nl = nl(x) iterated if iterate
    void stage(double a, double EoN2, const Vector &rhs, NonLinearState &nl, Vector &x, bool iterate);
    void prepareOutputState();
    bool step(double h, double &errNorm);
    void sample();
    void openOutput();
    void emitState(const std::string &jobFolder);
    void runPeriodic();
    void advancePeriod(double t0, double T);
    void writeEedfSamples();
    double horizon() const;
    /// gamma*(E/N)^2 * sum_k u_k du_k (F f)_k: power from the field per electron at unit density (eVm3/s)
    double fieldPower(const Vector &f, double EoN2) const;
    double meanEnergy(const Vector &f) const { return f.dot(m_u32du); }
    void resetAccumulators();
    void accumulate(double h, const Vector &fOld, double pfOld, double pgOld, double e2Old);
    void writeCycleAverage(unsigned periods, double t0, double wall);

    ElectronKineticsBoltzmann &m_ek;
    WorkingConditions &m_wc;
    TimeDependentOptions m_opt;
    std::unique_ptr<Waveform> m_waveform;
    const double m_N;
    Grid::Index m_n;

    Vector m_W;        // sqrt(u_k)/gamma
    Vector m_sqrtU;
    Vector m_numberWeights;  // sqrt(u_k) du_k, for the normalization
    Vector m_uDu;            // u_k du_k
    Vector m_u32du;          // u_k^(3/2) du_k
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
    double m_cPrev = 0.0, m_tPrev = -1.0;  // for NonLinear::extrapolate
    NonLinearState m_nlPrev;                // nonlinear coefficients at m_tPrev
    bool m_ee = false;
    double m_ne = 0.0, m_dlnneDt = 0.0;

    // time integrals (trapezoidal rule over accepted steps), used for period averages
    bool m_accumulate = false;
    Vector m_accF;
    double m_accT = 0, m_accC = 0, m_accPfield = 0, m_accPgrowth = 0, m_accE2 = 0;
    std::size_t m_nextSample = 0;
    std::vector<double> m_extraStops;  // landing points besides samples and breakpoints (periodic phase grid)
    bool m_recordSchedule = false, m_frozen = false;
    std::vector<double> m_schedule;    // accepted step end times of a period, relative to its start
    double m_periodStart = 0.0;
    double m_hPeriodStart = 0.0;  // first step of every period (periodic mode)
    bool m_recordFirstStep = false;
    Diagnostics m_diag;

    std::filesystem::path m_outFolder;
    std::ofstream m_timeSeries, m_rates;
    std::vector<double> m_eedfTimes, m_eedfFields;
    std::vector<Vector> m_eedfSamples;
};

} // namespace loki

#endif // LOKI_CPP_TIMEDEPENDENT_H
