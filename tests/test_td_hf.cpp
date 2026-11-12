/** \file
 *
 *  Tests of the time-dependent HF mode (timeDependent.highFrequency): reducedField is the RMS envelope of a carrier at
 *  workingConditions.excitationFrequency, and every field term takes LoKI-B's steady HF form at that omega.
 *
 *  He (input/Helium/He_LXCat.txt), 0.5 Torr, fixed grid (0-60 eV, 600 cells), conservative ionization.
 *
 *  1. A constant envelope stays at LoKI-B's steady HF solution: started from it (initialCondition steadyState, which
 *     LoKI-B's steady solver computes with the HF field operator), f0 after 2 us equals it within 1e-8, for dynamic
 *     and quasi-static f1, at 100 MHz and 1 GHz.
 *  2. From a 0.5 eV Maxwellian the same envelope relaxes to that steady HF solution (5 us, 1e-6).
 *  3. The test is not empty: the 1 GHz steady HF EEDF differs from the DC one at the same E_rms (mean energy lower by
 *     more than 20 %), and highFrequency with excitationFrequency 0 equals the DC run to round-off.
 *  4. excitationFrequency != 0 without highFrequency is refused, as before.
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
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{

const double umax = 60.0;
const int cells = 600;

fs::path setup(const fs::path &dir, const fs::path &input, const std::string &name, double EoN, double freq,
               bool hf, const std::string &anisotropy, const std::string &initial)
{
    std::ostringstream s;
    s << "workingConditions:\n  reducedField: " << EoN << "\n  electronTemperature: 1\n  excitationFrequency: " << freq
      << "\n  gasPressure: 66.661184\n  gasTemperature: 300\n  electronDensity: 1e15\n  chamberLength: 1.0\n"
      << "  chamberRadius: 1.0\nelectronKinetics:\n  isOn: true\n  eedfType: boltzmann\n"
      << "  ionizationOperatorType: conservative\n  growthModelType: temporal\n  includeEECollisions: false\n"
      << "  LXCatFiles:\n    - " << (input / "Helium" / "He_LXCat.txt").string() << "\n  gasProperties:\n    mass: "
      << (input / "Databases" / "masses.txt").string() << "\n    fraction:\n      - He = 1\n"
      << "  stateProperties:\n    population:\n      - He(1S1) = 1.0\n"
      << "  timeDependent:\n    isOn: true\n    reducedField: " << EoN << "\n    anisotropy: " << anisotropy << "\n"
      << "    highFrequency: " << (hf ? "true" : "false") << "\n    initialCondition:\n" << initial
      << "    finalTime: 1e-5\n"
      << "  numerics:\n    energyGrid:\n      maxEnergy: " << umax << "\n      cellNumber: " << cells << "\n"
      << "    maxPowerBalanceRelError: 1e-9\n    nonLinearRoutines:\n      algorithm: mixingDirectSolutions\n"
      << "      mixingParameter: 0.7\n      maxEedfRelError: 1e-9\n"
      << "chemistry:\n  isOn: false\ngui:\n  isOn: false\noutput:\n  isOn: false\n  folder: td_test\n";
    const fs::path in = dir / name;
    std::ofstream(in) << s.str();
    return in;
}

std::string steady(double EoN)
{
    std::ostringstream s;
    s << "      type: steadyState\n      reducedField: " << EoN << "\n";
    return s.str();
}

const std::string maxwellian = "      type: maxwellian\n      temperature: 0.5\n";

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

double relMax(const loki::Vector &a, const loki::Vector &b)
{
    return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff();
}

} // namespace

int main()
{
    const fs::path input = fs::path(loki::getEnvironmentVariable("LOKI_TEST_INPUT_DIR")) / ".." / "input";
    const fs::path dir = fs::temp_directory_path() / "loki_test_td_hf";
    fs::create_directories(dir);

    // 1, 2: a constant envelope is LoKI-B's steady HF solution
    for (const double freq : {1e8, 1e9})
        for (const std::string an : {"dynamic", "quasiStatic"})
        {
            const double E = freq > 5e8 ? 100.0 : 30.0;
            const std::string tag = an + "_" + std::to_string(int(freq / 1e6));
            Run a = make(setup(dir, input, "s_" + tag + ".in", E, freq, true, an, steady(E)));
            a.td->initialize();
            const loki::Vector f0 = a.td->eedf();
            a.td->advance(2e-6);
            const double d1 = relMax(a.td->eedf(), f0);
            Run b = make(setup(dir, input, "m_" + tag + ".in", E, freq, true, an, maxwellian));
            b.td->initialize();
            b.td->advance(5e-6);
            const double d2 = relMax(b.td->eedf(), f0);
            std::clog << tag << " MHz, E_rms " << E << " Td: <eps> steady HF " << a.td->meanEnergy()
                      << " eV; from it after 2 us " << d1 << ", from a Maxwellian after 5 us " << d2 << std::endl;
            test_expr(d1 < 1e-8);
            test_expr(d2 < 1e-6);
        }

    // 3. not empty: HF differs from DC at the same E_rms; highFrequency at f = 0 is the DC run
    {
        Run hf = make(setup(dir, input, "hf.in", 100.0, 1e9, true, "dynamic", steady(100.0)));
        Run dc = make(setup(dir, input, "dc.in", 100.0, 0.0, false, "dynamic", steady(100.0)));
        Run dc0 = make(setup(dir, input, "dc0.in", 100.0, 0.0, true, "dynamic", steady(100.0)));
        hf.td->initialize();
        dc.td->initialize();
        dc0.td->initialize();
        hf.td->advance(1e-7);
        dc.td->advance(1e-7);
        dc0.td->advance(1e-7);
        std::clog << "100 Td: <eps> 1 GHz HF " << hf.td->meanEnergy() << " eV, DC " << dc.td->meanEnergy()
                  << " eV; highFrequency at f = 0 vs DC " << relMax(dc0.td->eedf(), dc.td->eedf()) << std::endl;
        test_expr(hf.td->meanEnergy() < 0.8 * dc.td->meanEnergy());
        test_expr(relMax(dc0.td->eedf(), dc.td->eedf()) == 0.0);
    }

    // 4. refused without highFrequency
    {
        bool threw = false;
        try
        {
            Run r = make(setup(dir, input, "refuse.in", 30.0, 1e9, false, "dynamic", steady(30.0)));
        }
        catch (const std::exception &e)
        {
            threw = std::string(e.what()).find("highFrequency") != std::string::npos;
        }
        test_expr(threw);
    }
    fs::remove_all(dir);
    test_report;
    return nerrors;
}
