/** \file
 *
 *  Tests of TimeDependentBoltzmann::setFieldScale and the warm start with f1, initialize(u, f0, uFaces, f1): the
 *  hooks of a coupled model's power control (the field rescaled by a gain while the solver runs) and of a restarted
 *  solver (regridding, a redone step) that must continue the electrons' momentum state.
 *
 *  He (input/Helium/He_LXCat.txt), 0.5 Torr, dynamic f1, fixed grid (0-60 eV, 600 cells), from a 0.5 eV Maxwellian
 *  (a gas-temperature Maxwellian is not resolved by 0.1 eV cells: f0 < 0 in the first cell, which a warm start's
 *  log-linear interpolation drops). At 0.5 Torr f1 relaxes within ~1 ns, f0 heats over ~100 ns.
 *
 *  1. setFieldScale(2) before initialize() on a 20 Td run = a 40 Td run (same initial f1 and steps: equal to
 *     round-off). (After initialize() it is a field jump at t = 0, which f1 follows within ~1/nu_m.)
 *  2. setFieldScale(2) at t = 1 ns on a 20 Td run = a 20 Td run to 1 ns continued by a 40 Td solver warm-started
 *     from its f0 and f1 (mean energy and f1 within the integrator's tolerance).
 *  3. A 40 Td run to 2 ns restarted every 0.1 ns from its own state: with f1 carried, the mean energy equals the
 *     continuous run's; with f0 only (f1 reset to the quasi-static f1) the electrons overheat (the 2026-10-29 bug in
 *     the coupled power drive: tens of eV within ns).
 */

#include "LoKI-B/Environment.h"
#include "LoKI-B/LegacyToJSON.h"
#include "LoKI-B/Simulation.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;

namespace
{

const double umax = 60.0;
const int cells = 600;

fs::path setup(const fs::path &dir, const fs::path &input, const std::string &name, double EoN)
{
    std::ostringstream s;
    s << "workingConditions:\n  reducedField: " << EoN << "\n  electronTemperature: 1\n  excitationFrequency: 0\n"
      << "  gasPressure: 66.661184\n  gasTemperature: 300\n  electronDensity: 1e15\n  chamberLength: 1.0\n"
      << "  chamberRadius: 1.0\nelectronKinetics:\n  isOn: true\n  eedfType: boltzmann\n"
      << "  ionizationOperatorType: conservative\n  growthModelType: temporal\n  includeEECollisions: false\n"
      << "  LXCatFiles:\n    - " << (input / "Helium" / "He_LXCat.txt").string() << "\n  gasProperties:\n    mass: "
      << (input / "Databases" / "masses.txt").string() << "\n    fraction:\n      - He = 1\n"
      << "  stateProperties:\n    population:\n      - He(1S1) = 1.0\n"
      << "  timeDependent:\n    isOn: true\n    reducedField: " << EoN << "\n    anisotropy: dynamic\n"
      << "    initialCondition:\n      type: maxwellian\n      temperature: 0.5\n    finalTime: 1e-6\n"
      << "  numerics:\n    energyGrid:\n      maxEnergy: " << umax << "\n      cellNumber: " << cells << "\n"
      << "    maxPowerBalanceRelError: 1e-9\n    nonLinearRoutines:\n      algorithm: mixingDirectSolutions\n"
      << "      mixingParameter: 0.7\n      maxEedfRelError: 1e-9\n"
      << "chemistry:\n  isOn: false\ngui:\n  isOn: false\noutput:\n  isOn: false\n  folder: td_test\n";
    const fs::path in = dir / name;
    std::ofstream(in) << s.str();
    return in;
}

struct Run
{
    std::unique_ptr<loki::Simulation> sim;
    loki::TimeDependentBoltzmann *td;
};

Run make(const fs::path &in)
{
    Run r{std::make_unique<loki::Simulation>(in, loki::legacyToJSON(in)), nullptr};
    r.td = r.sim->timeDependent();
    test_expr(r.td != nullptr);
    return r;
}

loki::Vector cellEnergies()
{
    loki::Vector u(cells);
    for (int k = 0; k < cells; ++k)
        u[k] = (k + 0.5) * umax / cells;
    return u;
}

loki::Vector faceEnergies()
{
    loki::Vector u(cells + 1);
    for (int k = 0; k <= cells; ++k)
        u[k] = k * umax / cells;
    return u;
}

double relMax(const loki::Vector &a, const loki::Vector &b)
{
    return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff();
}

} // namespace

int main()
{
    const fs::path input = fs::path(loki::getEnvironmentVariable("LOKI_TEST_INPUT_DIR")) / ".." / "input";
    const fs::path dir = fs::temp_directory_path() / "loki_test_td_field_scale";
    fs::create_directories(dir);
    const auto in20 = setup(dir, input, "e20.in", 20.0), in40 = setup(dir, input, "e40.in", 40.0);
    const loki::Vector u = cellEnergies(), uf = faceEnergies();

    // 1. scale before the start = the doubled field
    {
        Run a = make(in20), b = make(in40);
        a.td->setFieldScale(2.0);
        a.td->initialize();
        test_expr(a.td->fieldScale() == 2.0);
        b.td->initialize();
        a.td->advance(2e-9);
        b.td->advance(2e-9);
        std::clog << "scale at t = 0: <eps> " << a.td->meanEnergy() << " vs " << b.td->meanEnergy() << " eV"
                  << std::endl;
        test_expr(std::abs(a.td->meanEnergy() / b.td->meanEnergy() - 1) < 1e-12);
        test_expr(relMax(a.td->firstAnisotropyNodes(), b.td->firstAnisotropyNodes()) < 1e-12);
    }
    // 2. scale mid-run = a 40 Td solver warm-started from the 20 Td state (f0 and f1)
    {
        Run a = make(in20), r = make(in20), b = make(in40);
        a.td->initialize();
        a.td->advance(1e-9);
        a.td->setFieldScale(2.0);
        a.td->advance(3e-9);
        r.td->initialize();
        r.td->advance(1e-9);
        b.td->initialize(u, r.td->eedf(), uf, r.td->firstAnisotropyNodes());
        b.td->advance(2e-9);
        const double ea = a.td->meanEnergy(), eb = b.td->meanEnergy();
        const double df1 = relMax(a.td->firstAnisotropyNodes(), b.td->firstAnisotropyNodes());
        std::clog << "scale at 1 ns: <eps> " << ea << " vs warm-started 40 Td solver " << eb << " eV, f1 within "
                  << df1 << std::endl;
        test_expr(std::abs(ea / eb - 1) < 1e-3);
        test_expr(df1 < 1e-2);
    }
    // 3. restarts every 0.1 ns: with f1 carried = continuous; f0 only overheats
    {
        Run c = make(in40);
        c.td->initialize();
        c.td->advance(2e-9);
        const double ec = c.td->meanEnergy();
        double ew = 0, ef0 = 0;
        for (const bool carry : {true, false})
        {
            loki::Vector f0, f1;
            {
                Run s = make(in40);
                s.td->initialize();
                s.td->advance(1e-10);
                f0 = s.td->eedf();
                f1 = s.td->firstAnisotropyNodes();
            }
            for (int k = 1; k < 20; ++k)
            {
                Run s = make(in40);
                if (carry)
                    s.td->initialize(u, f0, uf, f1);
                else
                    s.td->initialize(u, f0);
                s.td->advance(1e-10);
                f0 = s.td->eedf();
                f1 = s.td->firstAnisotropyNodes();
                (carry ? ew : ef0) = s.td->meanEnergy();
            }
        }
        std::clog << "2 ns at 40 Td: continuous <eps> " << ec << " eV; restarted every 0.1 ns with f1 " << ew
                  << " eV, with f0 only " << ef0 << " eV" << std::endl;
        test_expr(std::abs(ew / ec - 1) < 5e-3);
        test_expr(ef0 / ec - 1 > 0.1);
    }
    fs::remove_all(dir);
    test_report;
    return nerrors;
}
