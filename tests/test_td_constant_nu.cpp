/** \file
 *
 *  Test of the time-dependent mode (TimeDependentBoltzmann) against exact solutions.
 *
 *  Model gas X (He mass, elastic only) with a constant collision frequency nu = N*nu0. For this gas the two-term
 *  equation keeps a Maxwellian Maxwellian, and the mean energy obeys exactly
 *      d<eps>/dt = P(t) - nu_u (<eps> - 3/2 kTg),  P = e (E/N)^2 N / (m nu0),  nu_u = (2m/M) N nu0.
 *  1. Field step 3 -> 5 Td from the steady state at 3 Td: <eps> at t = 1/nu_u (mid relaxation).
 *  2. E/N = 5 sin(omega t) Td, 2 omega = nu_u, periodic steady state: cycle-averaged <eps> = 3/2 kTg + P0/(2 nu_u).
 *  The tolerance (3e-3) covers the energy discretization of the steady operators (1e-3 on this grid).
 */

#include "LoKI-B/Constant.h"
#include "LoKI-B/LegacyToJSON.h"
#include "LoKI-B/Simulation.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace
{

const double e = 1.602176634e-19, me = 9.1093837015e-31, kB = 1.380649e-23;
const double M = 6.646476882582080e-27, nu0 = 8e-14, Tg = 300.0, pPa = 1333.22368;
const double N = pPa / (kB * Tg), nuU = 2 * me / M * N * nu0, kTg = kB * Tg / e;

double P(double EN) { return e * std::pow(EN * 1e-21, 2) * N / (me * nu0); }

void writeGas(const fs::path &dir)
{
    std::ofstream x(dir / "X.txt");
    x << "Model elastic-only gas X, constant collision frequency.\n\nELASTIC\nX(X)\n " << std::scientific
      << std::setprecision(6) << me / M << "\nSPECIES: e / X\nPROCESS: E + X(X) -> E + X(X), Elastic\nPARAM.:  m/M = " << me / M << "\n"
      << "COMMENT: [e + X(X) -> e + X(X), Elastic] model\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
      << "-----------------------------\n";
    for (int i = 0; i <= 700; ++i)
    {
        const double u = i == 0 ? 0.0 : 1e-5 * std::pow(1e9, (i - 1) / 699.0);
        x << ' ' << u << ' ' << nu0 / std::sqrt(2 * e * std::max(u, 1e-5) / me) << '\n';
    }
    x << "-----------------------------\n";
    std::ofstream(dir / "masses.txt") << "X    " << std::setprecision(16) << M << "\n";
}

fs::path setup(const fs::path &dir, double EN, const std::string &td)
{
    std::ostringstream s;
    s << "workingConditions:\n  reducedField: " << EN << "\n  electronTemperature: 1\n  excitationFrequency: 0\n"
      << "  gasPressure: " << std::setprecision(12) << pPa << "\n  gasTemperature: " << Tg
      << "\n  electronDensity: 1e19\n  chamberLength: 1.0\n  chamberRadius: 1.0\n"
      << "electronKinetics:\n  isOn: true\n  eedfType: boltzmann\n  ionizationOperatorType: conservative\n"
      << "  growthModelType: temporal\n  includeEECollisions: false\n  LXCatFiles:\n    - " << (dir / "X.txt").string()
      << "\n  gasProperties:\n    mass: " << (dir / "masses.txt").string() << "\n    fraction:\n      - X = 1\n"
      << "  stateProperties:\n    population:\n      - X(X) = 1.0\n" << td
      << "  numerics:\n    energyGrid:\n      maxEnergy: 60\n      cellNumber: 1500\n"
      << "    maxPowerBalanceRelError: 1e-9\n    nonLinearRoutines:\n      algorithm: mixingDirectSolutions\n"
      << "      mixingParameter: 0.7\n      maxEedfRelError: 1e-9\n"
      << "chemistry:\n  isOn: false\ngui:\n  isOn: false\noutput:\n  isOn: false\n  folder: td_test\n"
      << "  dataFiles:\n    - eedf\n";
    const fs::path in = dir / "setup.in";
    std::ofstream(in) << s.str();
    return in;
}

/// run and return the mean energy for each job folder emitted (td_initial, td_final, td_cycleAverage)
std::map<std::string, double> run(const fs::path &input)
{
    std::map<std::string, double> out;
    const auto cnf = loki::legacyToJSON(input);
    loki::Simulation sim(input, cnf);
    sim.obtainedResults().addListener(
        [&out](const loki::Grid &, const loki::Vector &, const loki::WorkingConditions &wc, const loki::Power &,
               const loki::EedfCollisionDataMixture &, const loki::SwarmParameters &sp, const loki::Vector *) {
            out[wc.getCurrentJobFolder()] = sp.meanEnergy;
        });
    sim.run();
    return out;
}

} // namespace

int main()
{
    const fs::path dir = fs::temp_directory_path() / "loki_test_td_constant_nu";
    fs::create_directories(dir);
    writeGas(dir);

    // 1. step 3 -> 5 Td, compare at t = 1/nu_u
    {
        std::ostringstream td;
        td << "  timeDependent:\n    isOn: true\n    reducedField: 5\n    initialCondition:\n      type: steadyState\n"
           << "      reducedField: 3\n    finalTime: " << std::setprecision(12) << 1 / nuU << "\n";
        const auto r = run(setup(dir, 5, td.str()));
        const double eps0 = r.at("td_initial");
        const double einf = 1.5 * kTg + P(5) / nuU;
        const double exact = einf + (eps0 - einf) * std::exp(-1.0);
        std::clog << "step: <eps>(1/nu_u) = " << r.at("td_final") << " eV, exact " << exact << " eV" << std::endl;
        test_expr(std::abs(r.at("td_final") / exact - 1) < 3e-3);
        test_expr(std::abs(eps0 / (1.5 * kTg + P(3) / nuU) - 1) < 3e-3);
    }
    // 2. sinusoid, 2 omega = nu_u, periodic steady state
    {
        const double freq = nuU / (4 * loki::Constant::pi);
        std::ostringstream td;
        td << "  timeDependent:\n    isOn: true\n    reducedField:\n      type: sinusoid\n      amplitude: 5\n"
           << "      frequency: " << std::setprecision(12) << freq << "\n    initialCondition:\n      type: steadyState\n"
           << "      reducedField: rms\n    periodic:\n      tol: 1e-6\n";
        const auto r = run(setup(dir, 5, td.str()));
        const double exact = 1.5 * kTg + P(5) / (2 * nuU);
        std::clog << "sinusoid: cycle-averaged <eps> = " << r.at("td_cycleAverage") << " eV, exact " << exact << " eV"
                  << std::endl;
        test_expr(r.count("td_cycleAverage") == 1);
        test_expr(std::abs(r.at("td_cycleAverage") / exact - 1) < 3e-3);
    }
    fs::remove_all(dir);
    test_report;
    return nerrors;
}
