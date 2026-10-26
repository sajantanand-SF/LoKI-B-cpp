/** \file
 *
 *  Python bindings (pybind11) for LoKI-B++: steady runs through loki::Simulation and the time-dependent mode
 *  (loki::TimeDependentBoltzmann), in-process. Built with -DLOKIB_PYTHON=ON.
 *
 *  import lokib
 *  sim = lokib.Simulation(setup_text)          # LoKI-B legacy input text (format="legacy") or JSON (format="json")
 *  sim.run()                                   # steady runs (or the time-dependent run of the input); returns the
 *                                              # results emitted (one dict per E/N or output state)
 *  td = sim.time_dependent                     # None without electronKinetics.timeDependent.isOn
 *  td.initialize(); td.run_periodic(); td.periodic_result()
 *  td.initialize(u, f0)                        # warm start from an EEDF
 *  td.advance(t), td.set_populations({...}), td.set_electron_density(ne, dlnne_dt), td.rate_coefficients()
 *
 *  Rates are dicts {process description (as in LoKI's rateCoefficients.txt): (inelastic, superelastic)} in m3/s.
 *  Arrays are numpy copies. The GIL is released during the C++ calls that solve.
 */

#include "LoKI-B/LegacyToJSON.h"
#include "LoKI-B/Simulation.h"
#include "LoKI-B/TimeDependent.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace loki;

namespace
{

std::string describe(const EedfCollision &c)
{
    std::ostringstream os;
    os << c;
    return os.str();
}

py::dict ratesDict(const std::vector<RateCoefficient> &rates)
{
    py::dict d;
    for (const auto &rc : rates)
        d[py::str(describe(*rc.collision))] = py::make_tuple(rc.inelastic, rc.superelastic);
    return d;
}

py::dict powerDict(const Power &p)
{
    py::dict d;
    d["field"] = p.field;
    d["elastic_net"] = p.elasticNet;
    d["car_net"] = p.carNet;
    d["inelastic"] = p.inelastic;
    d["superelastic"] = p.superelastic;
    d["excitation_net"] = p.excitation.net();
    d["ionization_net"] = p.ionization.net();
    d["electron_density_growth"] = p.eDensGrowth;
    d["electron_electron_net"] = p.electronElectronNet;
    d["balance"] = p.balance;
    d["relative_balance"] = p.relativeBalance;
    return d;
}

py::dict swarmDict(const SwarmParameters &s)
{
    py::dict d;
    d["mean_energy"] = s.meanEnergy;
    d["reduced_mobility"] = s.redMobCoeff;
    d["reduced_diffusion"] = s.redDiffCoeff;
    d["reduced_townsend"] = s.redTownsendCoeff;
    d["reduced_attachment"] = s.redAttCoeff;
    d["characteristic_energy"] = s.characEnergy;
    d["Te"] = s.Te;
    d["drift_velocity"] = s.driftVelocity;
    d["reduced_mobility_HF"] = s.redMobilityHF;
    return d;
}

/// one result emitted by LoKI-B (steady solution or a time-dependent output state), stored without the GIL
struct Stored
{
    double reducedField, excitationFrequency;
    Vector energy, eedf;
    Power power;
    SwarmParameters swarm;
    std::vector<std::pair<std::string, std::pair<double, double>>> rates;
};

class PySimulation
{
public:
    PySimulation(const std::string &setup, const std::string &format, const std::string &basePath)
    {
        json_type cnf;
        if (format == "legacy")
        {
            std::istringstream is(setup);
            cnf = legacyToJSON(is);
        }
        else if (format == "json")
            cnf = json_type::parse(setup);
        else
            throw std::invalid_argument("format must be 'legacy' or 'json'");
        m_sim = std::make_unique<Simulation>(std::filesystem::path(basePath), cnf);
        m_sim->obtainedResults().addListener(
            [this](const Grid &grid, const Vector &f, const WorkingConditions &wc, const Power &p,
                   const EedfCollisionDataMixture &cd, const SwarmParameters &sp, const Vector *) {
                Stored s{wc.reducedField(), wc.excitationFrequency(), grid.getCells(), f, p, sp, {}};
                for (const auto &rc : cd.rateCoefficients())
                    s.rates.emplace_back(describe(*rc.collision), std::make_pair(rc.inelastic, rc.superelastic));
                m_stored.push_back(std::move(s));
            });
    }

    py::list run()
    {
        m_stored.clear();
        {
            py::gil_scoped_release release;
            m_sim->run();
        }
        py::list out;
        for (const auto &s : m_stored)
        {
            py::dict d;
            d["reduced_field"] = s.reducedField;
            d["excitation_frequency"] = s.excitationFrequency;
            d["energy"] = s.energy;
            d["eedf"] = s.eedf;
            d["power"] = powerDict(s.power);
            d["swarm"] = swarmDict(s.swarm);
            py::dict r;
            for (const auto &[name, v] : s.rates)
                r[py::str(name)] = py::make_tuple(v.first, v.second);
            d["rates"] = r;
            out.append(d);
        }
        return out;
    }

    TimeDependentBoltzmann *timeDependent() { return m_sim->timeDependent(); }

private:
    std::unique_ptr<Simulation> m_sim;
    std::vector<Stored> m_stored;
};

} // namespace

PYBIND11_MODULE(lokib, m)
{
    m.doc() = "LoKI-B++ (two-term Boltzmann solver) with the time-dependent mode, in-process";

    py::class_<TimeDependentBoltzmann>(m, "TimeDependent")
        .def("initialize", py::overload_cast<>(&TimeDependentBoltzmann::initialize),
             py::call_guard<py::gil_scoped_release>(), "build the grid, operators and the configured initial state")
        .def("initialize", py::overload_cast<const Vector &, const Vector &>(&TimeDependentBoltzmann::initialize),
             py::arg("u"), py::arg("f0"), py::call_guard<py::gil_scoped_release>(),
             "warm start: as initialize(), with f0 (on energies u, eV) as the initial EEDF")
        .def("advance", &TimeDependentBoltzmann::advance, py::arg("t_end"), py::call_guard<py::gil_scoped_release>())
        .def("run_periodic", &TimeDependentBoltzmann::runPeriodic, py::call_guard<py::gil_scoped_release>(),
             "march to the periodic steady state and average (after initialize)")
        .def("periodic_result",
             [](const TimeDependentBoltzmann &td) {
                 const auto &r = td.periodicResult();
                 py::dict d;
                 d["periods"] = r.periods;
                 d["converged"] = r.converged;
                 d["reduced_field_rms"] = r.reducedFieldRMS;
                 d["energy"] = r.energy;
                 d["eedf"] = r.eedf;
                 d["mean_energy"] = r.meanEnergy;
                 d["CIEff"] = r.CIEff;
                 d["power"] = powerDict(r.power);
                 d["swarm"] = swarmDict(r.swarm);
                 d["rates"] = ratesDict(r.rates);
                 d["eedf_end"] = r.eedfEnd;
                 d["wall_time"] = r.wallTime;
                 return d;
             })
        .def("set_populations", &TimeDependentBoltzmann::setPopulations, py::arg("populations"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_electron_density", &TimeDependentBoltzmann::setElectronDensity, py::arg("ne"),
             py::arg("dlnne_dt") = 0.0)
        .def("rate_coefficients", [](TimeDependentBoltzmann &td) { return ratesDict(td.rateCoefficients()); })
        .def_property_readonly("time", &TimeDependentBoltzmann::time)
        .def_property_readonly("eedf", [](const TimeDependentBoltzmann &td) { return Vector(td.eedf()); })
        .def_property_readonly("f1", [](const TimeDependentBoltzmann &td) { return Vector(td.firstAnisotropyNodes()); })
        .def_property_readonly("CIEff", &TimeDependentBoltzmann::CIEff)
        .def_property_readonly("electron_density", &TimeDependentBoltzmann::electronDensity)
        .def("mean_energy", py::overload_cast<>(&TimeDependentBoltzmann::meanEnergy, py::const_));

    py::class_<PySimulation>(m, "Simulation")
        .def(py::init<const std::string &, const std::string &, const std::string &>(), py::arg("setup"),
             py::arg("format") = "legacy", py::arg("base_path") = ".")
        .def("run", &PySimulation::run,
             "run the input (steady runs, or the whole time-dependent run); returns the emitted results")
        .def_property_readonly("time_dependent", &PySimulation::timeDependent, py::return_value_policy::reference_internal);
}
