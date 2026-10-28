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
#include <limits>
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
    /// for a sinusoid offset + amplitude sin(omega t + phase): true and its parameters (E/N in Td)
    virtual bool sinusoid(double &amplitude, double &omega, double &phase, double &offset) const { return false; }
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

    /** The first anisotropy f1 (on the cell faces, where the field flux lives):
     *  quasiStatic: f1 = -(E/N)/Omega_c df/du at every instant (LoKI-B MATLAB; valid for field changes slower than
     *               1/nu_m, i.e. omega << nu_m);
     *  dynamic:     f1 is an unknown with (1/(N gamma sqrt(u))) df1/dt + Omega_c f1 = -(E/N) df/du, integrated with
     *               f; it is eliminated per implicit stage, so the cost is that of the quasi-static scheme. For a
     *               sinusoid at omega >~ nu_m this gives the HF heating 1/(Omega_c + omega^2/(N^2 gamma^2 u Omega_c)).
     */
    enum class Anisotropy { quasiStatic, dynamic };
    Anisotropy anisotropy = Anisotropy::quasiStatic;

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
    /** Acceleration of the march to the periodic state (design note §14). none: plain marching. anderson: Anderson
     *  mixing on the one-period map x -> P_T(x), x = (f0, f1) at the period start, over the last andersonDepth
     *  periods, once the step schedule is frozen (the map is then fixed). Each period still integrates the map once;
     *  the next period starts from the mixed state instead of P_T(x). Mixing stops for good when the map residual
     *  max_i |P_T(x)_i - x_i| / (|P_T(x)_i| + periodicAtol max P_T(x)) (f1: relative to its own scale) is below
     *  andersonTol (default periodicTol/100); the convergence test then needs the usual plain (unmixed) periods.
     *  andersonTol is well below periodicTol because the plain test also bounds the remaining change,
     *  delta r/(1 - r) with r the slowest contraction (~0.98 at low pressure): mixing to periodicTol only would
     *  leave ~50x of it to plain marching (V12d).
     */
    enum class Accelerator { none, anderson };
    Accelerator accelerator = Accelerator::none;
    unsigned andersonDepth = 8;
    double andersonTol = 0.0;  // 0: periodicTol/100

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
    /** Linear solver for stage matrices with entries below the first sub-diagonal (superelastic collisions: gains
     *  from below, e.g. with metastable populations). automatic: GMRES preconditioned with the upper-Hessenberg part
     *  (solved with LinAlg::hessenberg, ~N^2), to a relative residual krylovTol, with dense LU as the fallback when
     *  it has not converged after krylovMaxIter iterations; lu: dense LU (~N^3). Matrices that are tridiagonal or
     *  upper Hessenberg always use TDMA or LinAlg::hessenberg.
     *  sparse: every non-tridiagonal stage matrix (upper Hessenberg or not) is kept as its tridiagonal part plus a
     *  sparse remainder (the inelastic/superelastic bands, ~1 % of N^2) and solved by GMRES preconditioned with the
     *  tridiagonal part (TDMA), to krylovTol; ~nnz per iteration instead of ~N^2 per solve. If GMRES has not
     *  converged after krylovMaxIter iterations, the dense matrix is built and solved as with automatic (counted in
     *  sparseFallbacks).
     */
    enum class LinearSolver { automatic, lu, sparse };
    LinearSolver linearSolver = LinearSolver::automatic;
    double krylovTol = 1e-14;
    unsigned krylovMaxIter = 30;
};

/** Anderson mixing (type II) for a fixed-point iteration x = g(x) (periodic accelerator anderson, design note §14).
 *  Residuals are weighted, r = w (g(x) - x); the next start is g - dG gamma, with gamma the least-squares fit of r
 *  by the last depth residual differences dR (dG: the map-value differences). If the residual grows above 10 times
 *  the smallest one seen, the history is dropped and the next start is the plain map value.
 */
class AndersonMixer
{
public:
    explicit AndersonMixer(unsigned depth) : m_depth(depth) {}
    void setWeights(const Vector &w) { m_w = w; }
    bool hasWeights() const { return m_w.size() > 0; }
    /// x: the start, g = g(x): returns max |r|, and the next start in next
    double update(const Vector &x, const Vector &g, Vector &next);
    unsigned long restarts() const { return m_restarts; }

private:
    unsigned m_depth;
    Vector m_w, m_rOld, m_gOld;
    std::vector<Vector> m_dR, m_dG;
    double m_residualMin = std::numeric_limits<double>::infinity();
    unsigned long m_restarts = 0;
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
    /** Warm start: as initialize(), but the initial f0 is the given EEDF (energies u in eV, values f0; interpolated
     *  log-linearly onto the current grid, zero above u.back(), normalized) instead of the steady or Maxwellian state.
     *  f1 (dynamic anisotropy) is the harmonic f1 of this f0 for a sinusoid, the quasi-static f1 otherwise.
     */
    void initialize(const Vector &u, const Vector &f0);
    /** March to the periodic steady state and average over averagePeriods periods (periodic waveforms; run() calls
     *  it after initialize()). The results are in periodicResult().
     */
    void runPeriodic();

    /// results of the last runPeriodic(): the cycle averages and the state at its end (for a warm start)
    struct PeriodicResult
    {
        unsigned periods = 0;
        unsigned long andersonSteps = 0;  // mixed periods (accelerator anderson)
        bool converged = false;
        double reducedFieldRMS = 0.0;  // Td
        Vector energy, eedf;           // cell energies (eV), cycle-averaged f0
        double meanEnergy = 0.0, CIEff = 0.0;
        Power power;                   // of the cycle-averaged f0; field and growth powers are time averages
        SwarmParameters swarm;         // LoKI's DC formulas on the cycle-averaged f0
        std::vector<RateCoefficient> rates;
        Vector eedfEnd;                // f0 at the end (the start of the next period)
        double wallTime = 0.0;
    };
    const PeriodicResult &periodicResult() const { return m_periodicResult; }
    /// advance the state to tEnd under the configured waveform
    void advance(double tEnd);

    double time() const { return m_t; }
    const Vector &eedf() const { return m_f; }
    double CIEff() const { return m_c; }
    double electronDensity() const { return m_ne; }
    double meanEnergy() const { return meanEnergy(m_f); }
    /// f1 on the cell faces (dynamic anisotropy; empty otherwise)
    const Vector &firstAnisotropyNodes() const { return m_f1; }

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
        unsigned long krylovIterations = 0, krylovFallbacks = 0, sparseFallbacks = 0;
        unsigned long andersonSteps = 0, andersonRestarts = 0;
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
    /// solve m_work x = b (x overwrites b) by GMRES, preconditioned with the upper-Hessenberg part of m_work
    void solveKrylov(Vector &b);
    /// solve m_stage x = b (x overwrites b) by GMRES, preconditioned with its tridiagonal part; dense if that fails
    void solveSparse(Vector &b);
    /// "\nsparseFallbacks <count>" for the summaries (linearSolver sparse), empty otherwise
    std::string sparseFallbacksLine() const;
    /// A0 f from the sparse form (tridiagonal part of A0 plus m_stage.R; linearSolver sparse)
    Vector sparseA0Product(const Vector &f) const;
    /// implicit stage: (W - a N A(nl)) x = rhs, with nl = nl(x) iterated if iterate
    void stage(double a, double EoN2, const Vector &rhs, NonLinearState &nl, Vector &x, bool iterate);
    void prepareOutputState();

    // dynamic anisotropy
    /// g_j = (u_j/3)/Omega_j on the faces (the field coefficient at E/N = 1), 0 on the boundary faces
    Vector fieldCoefficientNodes(double c) const;
    /// D_j = (f_j - f_{j-1})/du_j on the interior faces (0 on the boundary faces)
    Vector gradientNodes(const Vector &f) const;
    /// (phi_{k+1} - phi_k)/du_k
    Vector divergence(const Vector &phi) const;
    /// df1/dt = N gamma sqrt(u) (-Omega f1 - e D)
    Vector f1Derivative(const Vector &f, const Vector &f1, double e, double c) const;
    /// N*A*f with the field term from f1 (dynamic anisotropy); e = E/N (V m^2, signed)
    Vector applyNADynamic(const Vector &f, const Vector &f1, double e, const NonLinearState &nl) const;
    double fieldPowerDynamic(const Vector &f1, double e) const;
    double reducedFieldSI(double t) const;
    void stageDynamic(double a, double e, const Vector &rhs0, const Vector &r1, NonLinearState &nl, Vector &x0,
                      Vector &x1, bool iterate);
    void solveSystemDynamic(double a, double e, const Vector &r1, const NonLinearState &nl, const Vector &rhs0,
                            Vector &x0, Vector &x1);
    bool stepDynamic(double h, double &errNorm);
    double currentFieldPower() const;
    Vector quasiStaticF1(const Vector &f, double e, double c) const;
    /// steady EEDF with LoKI-B's HF field operator (RMS field ErmsSI, omega/N = WoN), starting from f
    Vector steadyStateHF(const Vector &f, double ErmsSI, double WoN);
    /// the harmonic (periodic) f1 at time t for f0 = f under E/N = A sin(omega t + phase) (A in V m^2)
    Vector harmonicF1(const Vector &f, double A, double omega, double phase, double t, double c) const;
    bool step(double h, double &errNorm);
    void sample();
    void openOutput();
    void emitState(const std::string &jobFolder);
    void initializeImpl(const Vector *u, const Vector *f0);
    void advancePeriod(double t0, double T);
    /// the period-start state x = (f0, f1) (f1 only with dynamic anisotropy), and its inverse (renormalizes f0)
    Vector periodState() const;
    void setPeriodState(const Vector &x);
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
    enum class Solver { tdma, hessenberg, lu, krylov, sparse } m_solver = Solver::lu;
    Matrix m_work;
    // linearSolver sparse: the stage matrix, the tridiagonal part of A0, and the dense solver for fallbacks
    TridiagonalPlusSparse m_stage;
    Vector m_A0lo, m_A0d, m_A0up;
    Solver m_denseSolver = Solver::krylov;
    bool m_denseCurrent = false;  // m_work holds the current stage matrix (after a fallback)
    Matrix m_krylovV, m_krylovZ;  // GMRES basis and preconditioned basis (N x krylovMaxIter+1)
    bool m_luCurrent = false;     // krylov: m_lu holds the factorization of m_work (after a fallback)
    LinAlg::HessenbergWorkspace m_hws;
    Eigen::PartialPivLU<Matrix> m_lu;

    Vector m_f;
    double m_t = 0.0;
    double m_c = 0.0;
    double m_h = 0.0;
    double m_cPrev = 0.0, m_tPrev = -1.0;  // for NonLinear::extrapolate
    NonLinearState m_nlPrev;                // nonlinear coefficients at m_tPrev
    bool m_ee = false;
    bool m_dynF1 = false;
    Vector m_f1;                     // f1 on the n+1 faces (dynamic anisotropy)
    Vector m_uNode, m_sqrtUNode, m_dn, m_dc;  // face energies, sqrt, face spacing (cell centre to centre), cell widths
    Vector m_f1Scale;                // per-face f1 magnitude for the error weights
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
    PeriodicResult m_periodicResult;

    std::filesystem::path m_outFolder;
    std::ofstream m_timeSeries, m_rates;
    std::vector<double> m_eedfTimes, m_eedfFields;
    std::vector<Vector> m_eedfSamples;
};

} // namespace loki

#endif // LOKI_CPP_TIMEDEPENDENT_H
