/** \file
 *
 *  Test of the coupling hook TimeDependentBoltzmann::setPopulations.
 *
 *  He (input/Helium/He_LXCat.txt, reversible excitations), 10 Torr, 20 Td, fixed grid. A time-dependent run starts
 *  with all atoms in He(1S1), relaxes, then the populations are changed to He(1S1) = 0.95, He(2S3) = 0.05 and the
 *  run continues at constant field. The final state must equal the steady LoKI-B solution with those populations
 *  (mean energy and the 2S3 excitation rate coefficient). In the bundled He file the excitations are irreversible
 *  (no superelastic term); He(2S3) gets a zero placeholder elastic cross section, so the populations act through the
 *  ground-state density (all collisions here start from He(1S1)).
 *
 *  Second case (RF): sinusoidal field (E_rms = 20 Td, 13.56 MHz), dynamic f1, equal sharing with the CIEff
 *  coefficients extrapolated in time. The populations change at t = 2 us; at t = 4 us (~14 bulk relaxation times
 *  later) the state must equal that of a run with the new populations from t = 0 (the extrapolation restarts at the
 *  change instead of extrapolating across it).
 *
 *  Third case (superelastic): as the first, with a copy of the He file in which 1S1 <-> 2S3 is reversible. With
 *  He(2S3) = 0 the stage matrix is upper Hessenberg; after the change to He(2S3) = 0.05 the superelastic gains from
 *  below (19.82 eV, ~250 cells under the diagonal) appear, so setPopulations must change the matrix structure (auto:
 *  Hessenberg -> Hessenberg-preconditioned GMRES; sparse: new entries in the sparse remainder). The final state must
 *  equal the steady LoKI-B solution with the new populations, including the superelastic rate coefficient.
 *
 *  All cases run with integrator.linearSolver auto and sparse (setPopulations must rebuild the sparse operator).
 */

#include "LoKI-B/Environment.h"
#include "LoKI-B/LegacyToJSON.h"
#include "LoKI-B/Simulation.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace
{

const std::string exc2S3 = "e + He(1S1) -> e + He(2S3), Excitation";
const std::string exc2S3rev = "e + He(1S1) <-> e + He(2S3), Excitation";

fs::path setup(const fs::path &dir, const fs::path &input, const std::string &name, const std::string &pops,
               const std::string &td, const std::string &sharing = "conservative", const fs::path &lxcat = {})
{
    std::ofstream(dir / "He2S3_elastic.txt")
        << "Placeholder: zero elastic cross section for He(2S3) (test only).\n\nELASTIC\nHe(2S3)\n 1.371400e-4\n"
        << "SPECIES: e / He\nPROCESS: E + He(2S3) -> E + He(2S3), Elastic\nPARAM.:  m/M = 0.00013714\n"
        << "COMMENT: [e + He(2S3) -> e + He(2S3), Elastic] placeholder\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
        << "-----------------------------\n 0.000000e+0 0.000000e+0\n 1.000000e+3 0.000000e+0\n"
        << "-----------------------------\n";
    std::ostringstream s;
    s << "workingConditions:\n  reducedField: 20\n  electronTemperature: 1\n  excitationFrequency: 0\n"
      << "  gasPressure: 1333.22368\n  gasTemperature: 300\n  electronDensity: 1e19\n  chamberLength: 1.0\n"
      << "  chamberRadius: 1.0\nelectronKinetics:\n  isOn: true\n  eedfType: boltzmann\n"
      << "  ionizationOperatorType: " << sharing << "\n  growthModelType: temporal\n  includeEECollisions: false\n"
      << "  LXCatFiles:\n    - " << (lxcat.empty() ? input / "Helium" / "He_LXCat.txt" : lxcat).string() << "\n    - "
      << (dir / "He2S3_elastic.txt").string() << "\n  gasProperties:\n    mass: "
      << (input / "Databases" / "masses.txt").string() << "\n    fraction:\n      - He = 1\n"
      << "  stateProperties:\n    statisticalWeight:\n      - He(1S1) = 1.0\n      - He(2S3) = 3.0\n"
      << "    population:\n" << pops << td
      << "  numerics:\n    energyGrid:\n      maxEnergy: 40\n      cellNumber: 500\n"
      << "    maxPowerBalanceRelError: 1e-9\n    nonLinearRoutines:\n      algorithm: mixingDirectSolutions\n"
      << "      mixingParameter: 0.7\n      maxEedfRelError: 1e-9\n"
      << "chemistry:\n  isOn: false\ngui:\n  isOn: false\noutput:\n  isOn: false\n  folder: td_test\n"
      << "  dataFiles:\n    - eedf\n";
    const fs::path in = dir / name;
    std::ofstream(in) << s.str();
    return in;
}

/// copy of the bundled He file with 1S1 <-> 2S3 reversible (LoKI-B reads the process from the COMMENT line)
fs::path reversibleHe(const fs::path &dir, const fs::path &input)
{
    std::ifstream in(input / "Helium" / "He_LXCat.txt");
    std::ostringstream s;
    s << in.rdbuf();
    std::string text = s.str();
    const std::string from = "[e + He(1S1) -> e + He(2S3), Excitation]";
    const auto pos = text.find(from);
    if (pos == std::string::npos)
        throw std::runtime_error("He_LXCat.txt: no 1S1 -> 2S3 excitation comment");
    text.replace(pos, from.size(), "[e + He(1S1) <-> e + He(2S3), Excitation]");
    const fs::path out = dir / "He_LXCat_2S3_reversible.txt";
    std::ofstream(out) << text;
    return out;
}

std::string describe(const loki::RateCoefficient &rc)
{
    std::ostringstream os;
    os << *rc.collision;
    return os.str();
}

} // namespace

int main()
{
    const fs::path input = fs::path(loki::getEnvironmentVariable("LOKI_TEST_INPUT_DIR")) / ".." / "input";
    const fs::path dir = fs::temp_directory_path() / "loki_test_td_set_populations";
    fs::create_directories(dir);
    const std::string popsB = "      - He(1S1) = 0.95\n      - He(2S3) = 0.05\n";

    // steady reference with the final populations
    double epsB = 0, kB = 0;
    {
        const auto in = setup(dir, input, "steady.in", popsB, "");
        loki::Simulation sim(in, loki::legacyToJSON(in));
        sim.obtainedResults().addListener(
            [&](const loki::Grid &, const loki::Vector &, const loki::WorkingConditions &, const loki::Power &,
                const loki::EedfCollisionDataMixture &cd, const loki::SwarmParameters &sp, const loki::Vector *) {
                epsB = sp.meanEnergy;
                for (const auto &rc : cd.rateCoefficients())
                    if (describe(rc) == exc2S3)
                        kB = rc.inelastic;
            });
        sim.run();
    }
    // steady reference with the final populations and the reversible 1S1 <-> 2S3 excitation
    const fs::path heRev = reversibleHe(dir, input);
    double epsS = 0, kS = 0, kSupS = 0;
    {
        const auto in = setup(dir, input, "steady_rev.in", popsB, "", "conservative", heRev);
        loki::Simulation sim(in, loki::legacyToJSON(in));
        sim.obtainedResults().addListener(
            [&](const loki::Grid &, const loki::Vector &, const loki::WorkingConditions &, const loki::Power &,
                const loki::EedfCollisionDataMixture &cd, const loki::SwarmParameters &sp, const loki::Vector *) {
                epsS = sp.meanEnergy;
                for (const auto &rc : cd.rateCoefficients())
                    if (describe(rc) == exc2S3rev)
                    {
                        kS = rc.inelastic;
                        kSupS = rc.superelastic;
                    }
            });
        sim.run();
    }
    for (const std::string solver : {"auto", "sparse"})
    {
    std::clog << "linearSolver " << solver << ":" << std::endl;
    // time-dependent: ground state only, then the populations of the steady run
    {
        const std::string td = "  timeDependent:\n    isOn: true\n    reducedField: 20\n    initialCondition:\n"
                               "      type: maxwellian\n    finalTime: 1e-5\n    integrator:\n      linearSolver: " +
                               solver + "\n";
        const auto in = setup(dir, input, "td.in", "      - He(1S1) = 1.0\n      - He(2S3) = 0.0\n", td);
        loki::Simulation sim(in, loki::legacyToJSON(in));
        loki::TimeDependentBoltzmann *tdb = sim.timeDependent();
        test_expr(tdb != nullptr);
        tdb->initialize();
        tdb->advance(2e-6);
        const double epsA = tdb->meanEnergy();
        tdb->setPopulations({{"He(1S1)", 0.95}, {"He(2S3)", 0.05}});
        tdb->advance(5e-6);
        double k = 0;
        for (const auto &rc : tdb->rateCoefficients())
            if (describe(rc) == exc2S3)
                k = rc.inelastic;
        std::clog << "ground state only: <eps> = " << epsA << " eV; after setPopulations: <eps> = " << tdb->meanEnergy()
                  << " eV (steady " << epsB << "), k(2S3) = " << k << " (" << kB << ")" << std::endl;
        test_expr(std::abs(epsA / epsB - 1) > 3e-3);  // the populations matter at this level
        test_expr(std::abs(tdb->meanEnergy() / epsB - 1) < 1e-6);
        test_expr(std::abs(k / kB - 1) < 1e-6);
    }
    // RF, dynamic f1, extrapolated CIEff: populations changed mid-run vs the new populations from the start
    {
        const std::string td = "  timeDependent:\n    isOn: true\n    reducedField:\n      type: sinusoid\n"
                               "      amplitude: 28.284271247\n      frequency: 13.56e6\n    anisotropy: dynamic\n"
                               "    initialCondition:\n      type: maxwellian\n    finalTime: 1e-5\n"
                               "    integrator:\n      nonLinear: extrapolate\n      linearSolver: " +
                               solver + "\n";
        const auto runRF = [&](const std::string &name, const std::string &pops, bool change) {
            const auto in = setup(dir, input, name, pops, td, "equalSharing");
            loki::Simulation sim(in, loki::legacyToJSON(in));
            loki::TimeDependentBoltzmann *tdb = sim.timeDependent();
            tdb->initialize();
            tdb->advance(2e-6);
            if (change)
                tdb->setPopulations({{"He(1S1)", 0.95}, {"He(2S3)", 0.05}});
            tdb->advance(4e-6);
            double k = 0;
            for (const auto &rc : tdb->rateCoefficients())
                if (describe(rc) == exc2S3)
                    k = rc.inelastic;
            return std::pair<double, double>{tdb->meanEnergy(), k};
        };
        const auto fresh = runRF("rf_fresh.in", popsB, false);
        const auto changed = runRF("rf_changed.in", "      - He(1S1) = 1.0\n      - He(2S3) = 0.0\n", true);
        std::clog << "RF at t = 4 us: <eps> = " << changed.first << " eV (populations from the start: " << fresh.first
                  << "), k(2S3) = " << changed.second << " (" << fresh.second << ")" << std::endl;
        test_expr(std::abs(changed.first / fresh.first - 1) < 1e-5);
        test_expr(std::abs(changed.second / fresh.second - 1) < 1e-4);
    }
    // superelastic: the populations change adds the gains from below
    {
        const std::string td = "  timeDependent:\n    isOn: true\n    reducedField: 20\n    initialCondition:\n"
                               "      type: maxwellian\n    finalTime: 1e-5\n    integrator:\n      linearSolver: " +
                               solver + "\n";
        const auto in = setup(dir, input, "td_rev.in", "      - He(1S1) = 1.0\n      - He(2S3) = 0.0\n", td,
                              "conservative", heRev);
        loki::Simulation sim(in, loki::legacyToJSON(in));
        loki::TimeDependentBoltzmann *tdb = sim.timeDependent();
        tdb->initialize();
        tdb->advance(2e-6);
        tdb->setPopulations({{"He(1S1)", 0.95}, {"He(2S3)", 0.05}});
        tdb->advance(5e-6);
        double k = 0, kSup = 0;
        for (const auto &rc : tdb->rateCoefficients())
            if (describe(rc) == exc2S3rev)
            {
                k = rc.inelastic;
                kSup = rc.superelastic;
            }
        std::clog << "superelastic after setPopulations: <eps> = " << tdb->meanEnergy() << " eV (steady " << epsS
                  << ", irreversible " << epsB << "), k(2S3) = " << k << " (" << kS << "), k_sup = " << kSup << " ("
                  << kSupS << ")" << std::endl;
        test_expr(std::abs(epsS / epsB - 1) > 1e-3);  // the superelastic gains matter at this level
        test_expr(kSupS > 0.0);
        test_expr(std::abs(tdb->meanEnergy() / epsS - 1) < 1e-6);
        test_expr(std::abs(k / kS - 1) < 1e-6);
        test_expr(std::abs(kSup / kSupS - 1) < 1e-6);
    }
    }
    fs::remove_all(dir);
    test_report;
    return nerrors;
}
