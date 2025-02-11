#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace fortran {
#include "rte_types.h"
#include "rte_kernels.h"
#include "rrtmgp_kernels.h"
}

using fortran::Float;
using fortran::Bool;

namespace py = pybind11;

PYBIND11_MODULE(pyrte_rrtmgp, m) {

    m.def("rte_lw_solver_noscat",
    [](
        int ncol,
        int nlay,
        int ngpt,
        bool top_at_1,
        int nmus,
        py::array_t<Float> Ds,
        py::array_t<Float> weights,
        py::array_t<Float> tau,
        py::array_t<Float> lay_source,
        py::array_t<Float> lev_source,
        py::array_t<Float> sfc_emis,
        py::array_t<Float> sfc_src,
        py::array_t<Float> inc_flux,
        py::array_t<Float> flux_up,
        py::array_t<Float> flux_dn,
        bool do_broadband,
        py::array_t<Float> broadband_up,
        py::array_t<Float> broadband_dn,
        bool do_Jacobians,
        py::array_t<Float> sfc_srcJac,
        py::array_t<Float> flux_upJac,
        bool do_rescaling,
        py::array_t<Float> ssa,
        py::array_t<Float> g
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmus <= 0) {
            throw std::runtime_error("ncol, nlay, ngpt, and nmus must be positive integers");
        }

        if (Ds.size()           != ncol * ngpt * nmus)       throw std::runtime_error("Invalid size for input array 'Ds': expected nmus elements.");
        if (weights.size()      != nmus)                     throw std::runtime_error("Invalid size for input array 'weights': expected nmus elements.");
        if (tau.size()          != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'tau': expected (ncol, nlay, ngpt).");
        if (lay_source.size()   != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'lay_source': expected (ncol, nlay, ngpt).");
        if (lev_source.size()   != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for input array 'lev_source': expected (ncol, nlay+1, ngpt).");
        if (sfc_emis.size()     != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_emis': expected (ncol, ngpt).");
        if (sfc_src.size()      != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_src': expected (ncol, ngpt).");
        if (inc_flux.size()     != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'inc_flux': expected (ncol, ngpt).");
        if (flux_up.size()      != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_up': expected (ncol, nlay+1, ngpt).");
        if (flux_dn.size()      != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_dn': expected (ncol, nlay+1, ngpt).");
        if (broadband_up.size() != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'broadband_up': expected (ncol, nlay+1).");
        if (broadband_dn.size() != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'broadband_dn': expected (ncol, nlay+1).");
        if (sfc_srcJac.size()   != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_srcJac': expected (ncol, ngpt).");
        if (flux_upJac.size()   != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'flux_upJac': expected (ncol, nlay+1).");
        if (ssa.size()          != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'ssa': expected (ncol, nlay, ngpt).");
        if (g.size()            != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'g': expected (ncol, nlay, ngpt).");

        int top_at_1_int = static_cast<int>(top_at_1);
        int do_broadband_int = static_cast<int>(do_broadband);
        int do_Jacobians_int = static_cast<int>(do_Jacobians);
        int do_rescaling_int = static_cast<int>(do_rescaling);

        py::buffer_info buf_Ds = Ds.request();
        py::buffer_info buf_weights = weights.request();
        py::buffer_info buf_tau = tau.request();
        py::buffer_info buf_lay_source = lay_source.request();
        py::buffer_info buf_lev_source = lev_source.request();
        py::buffer_info buf_sfc_emis = sfc_emis.request();
        py::buffer_info buf_sfc_src = sfc_src.request();
        py::buffer_info buf_inc_flux = inc_flux.request();
        py::buffer_info buf_flux_up = flux_up.request();
        py::buffer_info buf_flux_dn = flux_dn.request();
        py::buffer_info buf_broadband_up = broadband_up.request();
        py::buffer_info buf_broadband_dn = broadband_dn.request();
        py::buffer_info buf_sfc_srcJac = sfc_srcJac.request();
        py::buffer_info buf_flux_upJac = flux_upJac.request();
        py::buffer_info buf_ssa = ssa.request();
        py::buffer_info buf_g = g.request();

        fortran::rte_lw_solver_noscat(
            ncol,
            nlay,
            ngpt,
            top_at_1_int,
            nmus,
            reinterpret_cast<Float*>(buf_Ds.ptr),
            reinterpret_cast<Float*>(buf_weights.ptr),
            reinterpret_cast<Float*>(buf_tau.ptr),
            reinterpret_cast<Float*>(buf_lay_source.ptr),
            reinterpret_cast<Float*>(buf_lev_source.ptr),
            reinterpret_cast<Float*>(buf_sfc_emis.ptr),
            reinterpret_cast<Float*>(buf_sfc_src.ptr),
            reinterpret_cast<Float*>(buf_inc_flux.ptr),
            reinterpret_cast<Float*>(buf_flux_up.ptr),
            reinterpret_cast<Float*>(buf_flux_dn.ptr),
            do_broadband_int,
            reinterpret_cast<Float*>(buf_broadband_up.ptr),
            reinterpret_cast<Float*>(buf_broadband_dn.ptr),
            do_Jacobians_int,
            reinterpret_cast<Float*>(buf_sfc_srcJac.ptr),
            reinterpret_cast<Float*>(buf_flux_upJac.ptr),
            do_rescaling_int,
            reinterpret_cast<Float*>(buf_ssa.ptr),
            reinterpret_cast<Float*>(buf_g.ptr)
        );
    });


    m.def("rte_sw_solver_noscat",
    [](
        int ncol,
        int nlay,
        int ngpt,
        bool top_at_1,
        py::array_t<Float> tau,
        py::array_t<Float> mu0,
        py::array_t<Float> inc_flux_dir,
        py::array_t<Float> flux_dir
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau.size()          != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'tau': expected (ncol, nlay, ngpt).");
        if (mu0.size()          != ncol * nlay)              throw std::runtime_error("Invalid size for input array 'mu0': expected (ncol, nlay).");
        if (inc_flux_dir.size() != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'inc_flux_dir': expected (ncol, ngpt).");
        if (flux_dir.size()     != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_dir': expected (ncol, nlay+1, ngpt).");

        int top_at_1_int = static_cast<int>(top_at_1);

        py::buffer_info buf_tau = tau.request();
        py::buffer_info buf_mu0 = mu0.request();
        py::buffer_info buf_inc_flux_dir = inc_flux_dir.request();
        py::buffer_info buf_flux_dir = flux_dir.request();

        fortran::rte_sw_solver_noscat(
            ncol,
            nlay,
            ngpt,
            top_at_1_int,
            reinterpret_cast<Float*>(buf_tau.ptr),
            reinterpret_cast<Float*>(buf_mu0.ptr),
            reinterpret_cast<Float*>(buf_inc_flux_dir.ptr),
            reinterpret_cast<Float*>(buf_flux_dir.ptr)
        );
    });


    m.def("rte_sw_solver_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        bool top_at_1,
        py::array_t<Float> tau,
        py::array_t<Float> ssa,
        py::array_t<Float> g,
        py::array_t<Float> mu0,
        py::array_t<Float> sfc_alb_dir,
        py::array_t<Float> sfc_alb_dif,
        py::array_t<Float> inc_flux_dir,
        py::array_t<Float> flux_up,
        py::array_t<Float> flux_dn,
        py::array_t<Float> flux_dir,
        bool has_dif_bc,
        py::array_t<Float> inc_flux_dif,
        bool do_broadband,
        py::array_t<Float> broadband_up,
        py::array_t<Float> broadband_dn,
        py::array_t<Float> broadband_dir
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau.size()           != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'tau': expected (ncol, nlay, ngpt).");
        if (ssa.size()           != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'ssa': expected (ncol, nlay, ngpt).");
        if (g.size()             != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'g': expected (ncol, nlay, ngpt).");
        if (mu0.size()           != ncol * nlay)              throw std::runtime_error("Invalid size for input array 'mu0': expected (ncol, nlay).");
        if (sfc_alb_dir.size()   != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_alb_dir': expected (ncol, ngpt).");
        if (sfc_alb_dif.size()   != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_alb_dif': expected (ncol, ngpt).");
        if (inc_flux_dir.size()  != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'inc_flux_dir': expected (ncol, ngpt).");
        if (flux_up.size()       != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_up': expected (ncol, nlay+1, ngpt).");
        if (flux_dn.size()       != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_dn': expected (ncol, nlay+1, ngpt).");
        if (flux_dir.size()      != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_dir': expected (ncol, nlay+1, ngpt).");
        if (inc_flux_dif.size()  != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'inc_flux_dif': expected (ncol, ngpt).");
        if (broadband_up.size()  != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'broadband_up': expected (ncol, nlay+1).");
        if (broadband_dn.size()  != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'broadband_dn': expected (ncol, nlay+1).");
        if (broadband_dir.size() != ncol * (nlay + 1))        throw std::runtime_error("Invalid size for output array 'broadband_dir': expected (ncol, nlay+1).");

        int top_at_1_int = static_cast<int>(top_at_1);
        int has_dif_bc_int = static_cast<int>(has_dif_bc);
        int do_broadband_int = static_cast<int>(do_broadband);

        py::buffer_info buf_tau = tau.request();
        py::buffer_info buf_ssa = ssa.request();
        py::buffer_info buf_g = g.request();
        py::buffer_info buf_mu0 = mu0.request();
        py::buffer_info buf_sfc_alb_dir = sfc_alb_dir.request();
        py::buffer_info buf_sfc_alb_dif = sfc_alb_dif.request();
        py::buffer_info buf_inc_flux_dir = inc_flux_dir.request();
        py::buffer_info buf_flux_up = flux_up.request();
        py::buffer_info buf_flux_dn = flux_dn.request();
        py::buffer_info buf_flux_dir = flux_dir.request();
        py::buffer_info buf_inc_flux_dif = inc_flux_dif.request();
        py::buffer_info buf_broadband_up = broadband_up.request();
        py::buffer_info buf_broadband_dn = broadband_dn.request();
        py::buffer_info buf_broadband_dir = broadband_dir.request();

        fortran::rte_sw_solver_2stream(
            ncol,
            nlay,
            ngpt,
            top_at_1_int,
            reinterpret_cast<Float*>(buf_tau.ptr),
            reinterpret_cast<Float*>(buf_ssa.ptr),
            reinterpret_cast<Float*>(buf_g.ptr),
            reinterpret_cast<Float*>(buf_mu0.ptr),
            reinterpret_cast<Float*>(buf_sfc_alb_dir.ptr),
            reinterpret_cast<Float*>(buf_sfc_alb_dif.ptr),
            reinterpret_cast<Float*>(buf_inc_flux_dir.ptr),
            reinterpret_cast<Float*>(buf_flux_up.ptr),
            reinterpret_cast<Float*>(buf_flux_dn.ptr),
            reinterpret_cast<Float*>(buf_flux_dir.ptr),
            has_dif_bc_int,
            reinterpret_cast<Float*>(buf_inc_flux_dif.ptr),
            do_broadband_int,
            reinterpret_cast<Float*>(buf_broadband_up.ptr),
            reinterpret_cast<Float*>(buf_broadband_dn.ptr),
            reinterpret_cast<Float*>(buf_broadband_dir.ptr)
        );
    });


    m.def("rte_lw_solver_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        bool top_at_1,
        py::array_t<Float> tau,
        py::array_t<Float> ssa,
        py::array_t<Float> g,
        py::array_t<Float> lay_source,
        py::array_t<Float> lev_source,
        py::array_t<Float> sfc_emis,
        py::array_t<Float> sfc_src,
        py::array_t<Float> inc_flux,
        py::array_t<Float> flux_up,
        py::array_t<Float> flux_dn
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau.size()          != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'tau': expected (ncol, nlay, ngpt).");
        if (ssa.size()          != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'ssa': expected (ncol, nlay, ngpt).");
        if (g.size()            != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'g': expected (ncol, nlay, ngpt).");
        if (lay_source.size()   != ncol * nlay * ngpt)       throw std::runtime_error("Invalid size for input array 'lay_source': expected (ncol, nlay, ngpt).");
        if (lev_source.size()   != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for input array 'lev_source': expected (ncol, nlay+1, ngpt).");
        if (sfc_emis.size()     != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_emis': expected (ncol, ngpt).");
        if (sfc_src.size()      != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'sfc_src': expected (ncol, ngpt).");
        if (inc_flux.size()     != ncol * ngpt)              throw std::runtime_error("Invalid size for input array 'inc_flux': expected (ncol, ngpt).");
        if (flux_up.size()      != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_up': expected (ncol, nlay+1, ngpt).");
        if (flux_dn.size()      != ncol * (nlay + 1) * ngpt) throw std::runtime_error("Invalid size for output array 'flux_dn': expected (ncol, nlay+1, ngpt).");

        int top_at_1_int = static_cast<int>(top_at_1);

        py::buffer_info buf_tau = tau.request();
        py::buffer_info buf_ssa = ssa.request();
        py::buffer_info buf_g = g.request();
        py::buffer_info buf_lay_source = lay_source.request();
        py::buffer_info buf_lev_source = lev_source.request();
        py::buffer_info buf_sfc_emis = sfc_emis.request();
        py::buffer_info buf_sfc_src = sfc_src.request();
        py::buffer_info buf_inc_flux = inc_flux.request();
        py::buffer_info buf_flux_up = flux_up.request();
        py::buffer_info buf_flux_dn = flux_dn.request();

        fortran::rte_lw_solver_2stream(
            ncol,
            nlay,
            ngpt,
            top_at_1_int,
            reinterpret_cast<Float*>(buf_tau.ptr),
            reinterpret_cast<Float*>(buf_ssa.ptr),
            reinterpret_cast<Float*>(buf_g.ptr),
            reinterpret_cast<Float*>(buf_lay_source.ptr),
            reinterpret_cast<Float*>(buf_lev_source.ptr),
            reinterpret_cast<Float*>(buf_sfc_emis.ptr),
            reinterpret_cast<Float*>(buf_sfc_src.ptr),
            reinterpret_cast<Float*>(buf_inc_flux.ptr),
            reinterpret_cast<Float*>(buf_flux_up.ptr),
            reinterpret_cast<Float*>(buf_flux_dn.ptr)
        );
    });


    m.def("rte_increment_1scalar_by_1scalar",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau_inout.size() != ncol * nlay * ngpt ||
            tau_in.size() != ncol * nlay * ngpt
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();

        fortran::rte_increment_1scalar_by_1scalar(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr)
        );
    });


    m.def("rte_increment_1scalar_by_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();

        fortran::rte_increment_1scalar_by_2stream(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr)
        );
    });


    m.def("rte_increment_1scalar_by_nstream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();

        fortran::rte_increment_1scalar_by_nstream(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr)
        );
    });

    m.def("rte_increment_2stream_by_1scalar",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> tau_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();

        fortran::rte_increment_2stream_by_1scalar(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr)
        );
    });

    m.def("rte_increment_2stream_by_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> g_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (g_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt) ||
            (g_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_g_in = g_in.request();

        fortran::rte_increment_2stream_by_2stream(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_g_in.ptr)
        );
    });

    m.def("rte_increment_2stream_by_nstream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> p_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom <= 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nmom must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (g_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt) ||
            (p_in.size() != nmom * ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_p_in = p_in.request();

        fortran::rte_increment_2stream_by_nstream(
            ncol,
            nlay,
            ngpt,
            nmom,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_p_in.ptr)
        );
    });


    m.def("rte_increment_nstream_by_1scalar",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> tau_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay and ngpt must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();

        fortran::rte_increment_nstream_by_1scalar(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr)
        );
    });

    m.def("rte_increment_nstream_by_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom1,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> p_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> g_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom1 <= 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nmom1 must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (p_inout.size() != nmom1 * ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt) ||
            (g_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_p_inout = p_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_g_in = g_in.request();

        fortran::rte_increment_nstream_by_2stream(
            ncol,
            nlay,
            ngpt,
            nmom1,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_p_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_g_in.ptr)
        );
    });

    m.def("rte_increment_nstream_by_nstream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom1,
        int nmom2,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> p_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> p_in
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom1 <= 0 || nmom2 <= 0) {
            throw std::runtime_error("ncol, nlay, ngpt, nmom1 and nmom2 must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (p_inout.size() != nmom1 * ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt) ||
            (p_in.size() != nmom2 * ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_p_inout = p_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_p_in = p_in.request();

        fortran::rte_increment_nstream_by_nstream(
            ncol,
            nlay,
            ngpt,
            nmom1,
            nmom2,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_p_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_p_in.ptr)
        );
    });

    m.def("rte_inc_1scalar_by_1scalar_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_1scalar_by_1scalar_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<const Float*>(buf_tau_in.ptr),
            nbnd,
            reinterpret_cast<const int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_1scalar_by_2stream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (ssa_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_1scalar_by_2stream_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_1scalar_by_nstream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_1scalar_by_nstream_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });


    m.def("rte_inc_2stream_by_1scalar_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> tau_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_2stream_by_1scalar_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_2stream_by_2stream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> g_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (g_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (ssa_in.size() != ncol * nlay * nbnd) ||
            (g_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_g_in = g_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_2stream_by_2stream_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_g_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });


    m.def("rte_inc_2stream_by_nstream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> p_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom < 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt, nmom and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (g_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (ssa_in.size() != ncol * nlay * nbnd) ||
            (p_in.size() != nmom * ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_p_in = p_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_2stream_by_nstream_bybnd(
            ncol,
            nlay,
            ngpt,
            nmom,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_p_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_nstream_by_1scalar_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> tau_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_nstream_by_1scalar_bybnd(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_nstream_by_2stream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom1,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> p_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> g_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom1 <= 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt, nmom1 and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (p_inout.size() != nmom1 * ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (ssa_in.size() != ncol * nlay * nbnd) ||
            (g_in.size() != ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_p_inout = p_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_g_in = g_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_nstream_by_2stream_bybnd(
            ncol,
            nlay,
            ngpt,
            nmom1,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_p_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_g_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });

    m.def("rte_inc_nstream_by_nstream_bybnd",
    [](
        int ncol,
        int nlay,
        int ngpt,
        int nmom1,
        int nmom2,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> p_inout,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        py::array_t<Float> p_in,
        int nbnd,
        py::array_t<int> band_lims_gpoint
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || nmom1 <= 0 || nmom2 < 0 || nbnd < 0) {
            throw std::runtime_error("ncol, nlay, ngpt, nmom1, nmom2 and nbnd must be positive integers");
        }

        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (ssa_inout.size() != ncol * nlay * ngpt) ||
            (p_inout.size() != nmom1 * ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * nbnd) ||
            (ssa_in.size() != ncol * nlay * nbnd) ||
            (p_in.size() != nmom2 * ncol * nlay * nbnd) ||
            (band_lims_gpoint.size() != 2 * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_p_inout = p_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_p_in = p_in.request();
        py::buffer_info buf_band_lims_gpoint = band_lims_gpoint.request();

        fortran::rte_inc_nstream_by_nstream_bybnd(
            ncol,
            nlay,
            ngpt,
            nmom1,
            nmom2,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_p_inout.ptr),
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            reinterpret_cast<Float*>(buf_p_in.ptr),
            nbnd,
            reinterpret_cast<int*>(buf_band_lims_gpoint.ptr)
        );
    });


    m.def("rte_delta_scale_2str_k",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau_inout.size() != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'tau_inout': expected (ncol, nlay, ngpt).");
        if (ssa_inout.size() != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'ssa_inout': expected (ncol, nlay, ngpt).");
        if (g_inout.size()   != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'g_inout': expected (ncol, nlay, ngpt).");

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();

        fortran::rte_delta_scale_2str_k(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr)
        );
    });


    m.def("rte_delta_scale_2str_f_k",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_inout,
        py::array_t<Float> ssa_inout,
        py::array_t<Float> g_inout,
        py::array_t<Float> f
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau_inout.size() != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'tau_inout': expected (ncol, nlay, ngpt).");
        if (ssa_inout.size() != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'ssa_inout': expected (ncol, nlay, ngpt).");
        if (g_inout.size()   != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'g_inout': expected (ncol, nlay, ngpt).");
        if (f.size()         != ncol * nlay * ngpt) throw std::runtime_error("Invalid size for array 'f': expected (ncol, nlay, ngpt).");

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_ssa_inout = ssa_inout.request();
        py::buffer_info buf_g_inout = g_inout.request();
        py::buffer_info buf_f = f.request();

        fortran::rte_delta_scale_2str_f_k(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_inout.ptr),
            reinterpret_cast<Float*>(buf_ssa_inout.ptr),
            reinterpret_cast<Float*>(buf_g_inout.ptr),
            reinterpret_cast<Float*>(buf_f.ptr)
        );
    });


    m.def("rte_extract_subset_dim1_3d",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> array_in,
        int ncol_start,
        int ncol_end,
        py::array_t<Float> array_out
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || ncol_start < 0 || ncol_end < 0) {
            throw std::runtime_error("ncol, nlay, ngpt, ncol_start, and ncol_end must be positive integers");
        }

        if (array_in.size()  != ncol * nlay * ngpt)                        throw std::runtime_error("Invalid size for array 'array_in': expected (ncol, nlay, ngpt).");
        if (array_out.size() != (ncol_end - ncol_start + 1) * nlay * ngpt) throw std::runtime_error("Invalid size for array 'array_out': expected (ncol_end-ncol_start+1, nlay, ngpt).");

        py::buffer_info buf_array_in = array_in.request();
        py::buffer_info buf_array_out = array_out.request();

        fortran::rte_extract_subset_dim1_3d(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_array_in.ptr),
            ncol_start,
            ncol_end,
            reinterpret_cast<Float*>(buf_array_out.ptr)
        );
    });


    m.def("rte_extract_subset_dim2_4d",
    [](
        int nmom,
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> array_in,
        int ncol_start,
        int ncol_end,
        py::array_t<Float> array_out
    ) {
        if (nmom <= 0 || ncol <= 0 || nlay <= 0 || ngpt <= 0 || ncol_start < 0 || ncol_end < 0) {
            throw std::runtime_error("nmom, ncol, nlay, ngpt, ncol_start, and ncol_end must be positive integers");
        }

        if (array_in.size()  != nmom * ncol * nlay * ngpt)                        throw std::runtime_error("Invalid size for array 'array_in': expected (nmom, ncol, nlay, ngpt).");
        if (array_out.size() != nmom * (ncol_end - ncol_start + 1) * nlay * ngpt) throw std::runtime_error("Invalid size for array 'array_out': expected (nmom, ncol_end-ncol_start+1, nlay, ngpt).");

        py::buffer_info buf_array_in = array_in.request();
        py::buffer_info buf_array_out = array_out.request();

        fortran::rte_extract_subset_dim2_4d(
            nmom,
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_array_in.ptr),
            ncol_start,
            ncol_end,
            reinterpret_cast<Float*>(buf_array_out.ptr)
        );
    });


    m.def("rte_extract_subset_absorption_tau",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<Float> tau_in,
        py::array_t<Float> ssa_in,
        int ncol_start,
        int ncol_end,
        py::array_t<Float> tau_out
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0 || ncol_start < 0 || ncol_end < 0) {
            throw std::runtime_error("ncol, nlay, ngpt, ncol_start, and ncol_end must be positive integers");
        }

        if (tau_in.size()  != ncol * nlay * ngpt)                        throw std::runtime_error("Invalid size for array 'tau_in': expected (ncol, nlay, ngpt).");
        if (ssa_in.size()  != ncol * nlay * ngpt)                        throw std::runtime_error("Invalid size for array 'ssa_in': expected (ncol, nlay, ngpt).");
        if (tau_out.size() != (ncol_end - ncol_start + 1) * nlay * ngpt) throw std::runtime_error("Invalid size for array 'tau_out': expected (ncol_end-ncol_start+1, nlay, ngpt).");

        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();
        py::buffer_info buf_tau_out = tau_out.request();

        fortran::rte_extract_subset_absorption_tau(
            ncol,
            nlay,
            ngpt,
            reinterpret_cast<Float*>(buf_tau_in.ptr),
            reinterpret_cast<Float*>(buf_ssa_in.ptr),
            ncol_start,
            ncol_end,
            reinterpret_cast<Float*>(buf_tau_out.ptr)
        );
    });


    m.def("rte_sum_broadband",
    [](
        int ncol,
        int nlev,
        int ngpt,
        py::array_t<Float> gpt_flux,
        py::array_t<Float> flux
    ) {
        if (ncol <= 0 || nlev <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlev, and ngpt must be positive integers");
        }

        if (gpt_flux.size() != ncol * nlev * ngpt) throw std::runtime_error("Invalid size for input array 'gpt_flux': expected (ncol, nlev, ngpt).");
        if (flux.size()     != ncol * nlev)        throw std::runtime_error("Invalid size for output array 'flux': expected (ncol, nlev).");

        py::buffer_info buf_gpt_flux = gpt_flux.request();
        py::buffer_info buf_flux = flux.request();

        fortran::rte_sum_broadband(
            ncol,
            nlev,
            ngpt,
            reinterpret_cast<Float*>(buf_gpt_flux.ptr),
            reinterpret_cast<Float*>(buf_flux.ptr)
        );
    });

    m.def("rte_net_broadband_full",
    [](
        int ncol,
        int nlev,
        int ngpt,
        py::array_t<Float> gpt_flux_dn,
        py::array_t<Float> gpt_flux_up,
        py::array_t<Float> flux_net
    ) {
        if (ncol <= 0 || nlev <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlev, and ngpt must be positive integers");
        }

        if (gpt_flux_dn.size() != ncol * nlev * ngpt) throw std::runtime_error("Invalid size for input array 'gpt_flux_dn': expected (ncol, nlev, ngpt).");
        if (gpt_flux_up.size() != ncol * nlev * ngpt) throw std::runtime_error("Invalid size for input array 'gpt_flux_up': expected (ncol, nlev, ngpt).");
        if (flux_net.size()    != ncol * nlev)        throw std::runtime_error("Invalid size for output array 'flux_net': expected (ncol, nlev).");

        py::buffer_info buf_gpt_flux_dn = gpt_flux_dn.request();
        py::buffer_info buf_gpt_flux_up = gpt_flux_up.request();
        py::buffer_info buf_flux_net = flux_net.request();

        fortran::rte_net_broadband_full(
            ncol,
            nlev,
            ngpt,
            reinterpret_cast<Float*>(buf_gpt_flux_dn.ptr),
            reinterpret_cast<Float*>(buf_gpt_flux_up.ptr),
            reinterpret_cast<Float*>(buf_flux_net.ptr)
        );
    });

    m.def("rte_net_broadband_precalc",
    [](
        int ncol,
        int nlev,
        py::array_t<Float> broadband_flux_dn,
        py::array_t<Float> broadband_flux_up,
        py::array_t<Float> broadband_flux_net
    ) {
        if (ncol <= 0 || nlev <= 0) {
            throw std::runtime_error("ncol and nlev must be positive integers");
        }

        if (broadband_flux_dn.size()  != ncol * nlev) throw std::runtime_error("Invalid size for input array 'broadband_flux_dn': expected (ncol, nlev).");
        if (broadband_flux_up.size()  != ncol * nlev) throw std::runtime_error("Invalid size for input array 'broadband_flux_up': expected (ncol, nlev).");
        if (broadband_flux_net.size() != ncol * nlev) throw std::runtime_error("Invalid size for output array 'broadband_flux_net': expected (ncol, nlev).");

        py::buffer_info buf_broadband_flux_dn = broadband_flux_dn.request();
        py::buffer_info buf_broadband_flux_up = broadband_flux_up.request();
        py::buffer_info buf_broadband_flux_net = broadband_flux_net.request();

        fortran::rte_net_broadband_precalc(
            ncol,
            nlev,
            reinterpret_cast<Float*>(buf_broadband_flux_dn.ptr),
            reinterpret_cast<Float*>(buf_broadband_flux_up.ptr),
            reinterpret_cast<Float*>(buf_broadband_flux_net.ptr)
        );
    });

/// Disabled because they're not properly exported by Fortran
/// Probably a change in mo_fluxes_byband.F90 is required
#if 0
    m.def("rte_sum_byband",
    [](
        int ncol,
        int nlev,
        int ngpt,
        int nbnd,
        py::array_t<int> band_lims,
        py::array_t<Float> gpt_flux,
        py::array_t<Float> bnd_flux
    ) {
        if (ncol <= 0 || nlev <= 0 || ngpt <= 0 || nbnd <= 0) {
            throw std::runtime_error("ncol, nlev, ngpt and nbnd must be positive integers");
        }

        if (
            (band_lims.size() != 2 * nbnd) ||
            (gpt_flux.size() != ncol * nlev * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_band_lims = band_lims.request();
        py::buffer_info buf_gpt_flux = gpt_flux.request();
        py::buffer_info buf_bnd_flux = bnd_flux.request();

        fortran::rte_sum_byband(
            ncol,
            nlev,
            ngpt,
            nbnd,
            reinterpret_cast<int *>(buf_band_lims.ptr),
            reinterpret_cast<Float *>(buf_gpt_flux.ptr),
            reinterpret_cast<Float *>(buf_bnd_flux.ptr));
    });

    m.def("rte_net_byband_full",
    [](
        int ncol,
        int nlev,
        int ngpt,
        int nbnd,
        py::array_t<int> band_lims,
        py::array_t<Float> bnd_flux_dn,
        py::array_t<Float> bnd_flux_up,
        py::array_t<Float> bnd_flux_net
    ) {
        if (ncol <= 0 || nlev <= 0 || ngpt <= 0 || nbnd <= 0) {
            throw std::runtime_error("ncol, nlev, ngpt, nbnd and band_lims must be positive integers");
        }

        if (
            (band_lims.size() != 2 * nbnd) ||
            (bnd_flux_dn.size() != ncol * nlev * nbnd) ||
            (bnd_flux_up.size() != ncol * nlev * nbnd) ||
            (bnd_flux_net.size() != ncol * nlev * nbnd)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        py::buffer_info buf_band_lims = band_lims.request();
        py::buffer_info buf_bnd_flux_dn = bnd_flux_dn.request();
        py::buffer_info buf_bnd_flux_up = bnd_flux_up.request();
        py::buffer_info buf_bnd_flux_net = bnd_flux_net.request();

        fortran::rte_net_byband_full(
            ncol,
            nlev,
            ngpt,
            nbnd,
            reinterpret_cast<int *>(buf_band_lims.ptr),
            reinterpret_cast<Float *>(buf_bnd_flux_dn.ptr),
            reinterpret_cast<Float *>(buf_bnd_flux_up.ptr),
            reinterpret_cast<Float *>(buf_bnd_flux_net.ptr));
    });
#endif

    m.def("zero_array_1D", [](py::array_t<Float> arr) {
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.size >= INT_MAX) {
            throw std::runtime_error("Array size bigger than INT_MAX");
        }

        int ni = static_cast<int>(buf_info.size);
        Float* ptr = reinterpret_cast<Float*>(buf_info.ptr);

        fortran::zero_array_1D(ni, ptr);
    });


    m.def("zero_array_2D", [](py::array_t<Float> arr) {
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = static_cast<int>(buf_info.shape[0]);
        int nj = static_cast<int>(buf_info.shape[1]);
        Float* ptr = reinterpret_cast<Float*>(buf_info.ptr);

        fortran::zero_array_2D(ni, nj, ptr);
    });


    m.def("zero_array_3D", [](py::array_t<Float> arr) {
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 3) {
            throw std::runtime_error("Number of dimensions must be 3");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX || buf_info.shape[2] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = static_cast<int>(buf_info.shape[0]);
        int nj = static_cast<int>(buf_info.shape[1]);
        int nk = static_cast<int>(buf_info.shape[2]);
        Float* ptr = reinterpret_cast<Float*>(buf_info.ptr);

        fortran::zero_array_3D(ni, nj, nk, ptr);
    });


    m.def("zero_array_4D", [](py::array_t<Float> arr) {
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 4) {
            throw std::runtime_error("Number of dimensions must be 4");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX || buf_info.shape[2] >= INT_MAX || buf_info.shape[3] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = static_cast<int>(buf_info.shape[0]);
        int nj = static_cast<int>(buf_info.shape[1]);
        int nk = static_cast<int>(buf_info.shape[2]);
        int nl = static_cast<int>(buf_info.shape[3]);
        Float* ptr = reinterpret_cast<Float*>(buf_info.ptr);

        fortran::zero_array_4D(ni, nj, nk, nl, ptr);
    });


    /// RRTMGP methods

    m.def("rrtmgp_interpolation",
    [](
        int ncol,
        int nlay,
        int ngas,
        int nflav,
        int neta,
        int npres,
        int ntemp,
        py::array_t<int> flavor,
        py::array_t<Float> press_ref_log,
        py::array_t<Float> temp_ref,
        Float press_ref_log_delta,
        Float temp_ref_min,
        Float temp_ref_delta,
        Float press_ref_trop_log,
        py::array_t<Float> vmr_ref,
        py::array_t<Float> play,
        py::array_t<Float> tlay,
        py::array_t<Float> col_gas,
        py::array_t<int> jtemp,
        py::array_t<Float> fmajor,
        py::array_t<Float> fminor,
        py::array_t<Float> col_mix,
        py::array_t<Bool> tropo,
        py::array_t<int> jeta,
        py::array_t<int> jpress
    ) {
        if (ncol <= 0 || nlay <= 0 || ngas <= 0 || nflav <= 0 || neta <= 0 || npres <= 0 || ntemp <= 0) {
            throw std::runtime_error("ncol, nlay, ngas, nflav, neta, npres, and ntemp must be positive integers");
        }

        if (flavor.size()          != 2 * nflav)                       throw std::runtime_error("Invalid size for array 'flavor': expected (2, nflav).");
        if (press_ref_log.size()   != npres)                           throw std::runtime_error("Invalid size for array 'press_ref_log': expected (npres).");
        if (temp_ref.size()        != ntemp)                           throw std::runtime_error("Invalid size for array 'temp_ref': expected (ntemp).");
        if (vmr_ref.size()         != 2 * (ngas + 1) * ntemp)          throw std::runtime_error("Invalid size for array 'vmr_ref': expected (2, ngas+1, ntemp).");
        if (play.size()            != ncol * nlay)                     throw std::runtime_error("Invalid size for array 'play': expected (ncol, nlay).");
        if (tlay.size()            != ncol * nlay)                     throw std::runtime_error("Invalid size for array 'tlay': expected (ncol, nlay).");
        if (col_gas.size()         != ncol * nlay * (ngas + 1))        throw std::runtime_error("Invalid size for array 'col_gas': expected (ncol, nlay, ngas+1).");
        if (jtemp.size()           != ncol * nlay)                     throw std::runtime_error("Invalid size for array 'jtemp': expected (ncol, nlay).");
        if (fmajor.size()          != 2 * 2 * 2 * ncol * nlay * nflav) throw std::runtime_error("Invalid size for array 'fmajor': expected (2,2,2,ncol,nlay,nflav).");
        if (fminor.size()          != 2 * 2 * ncol * nlay * nflav)     throw std::runtime_error("Invalid size for array 'fminor': expected (2,2,ncol,nlay,nflav).");
        if (col_mix.size()         != 2 * ncol * nlay * nflav)         throw std::runtime_error("Invalid size for array 'col_mix': expected (2,ncol,nlay,nflav).");
        if (tropo.size()           != ncol * nlay)                     throw std::runtime_error("Invalid size for array 'tropo': expected (ncol, nlay).");
        if (jeta.size()            != 2 * ncol * nlay * nflav)         throw std::runtime_error("Invalid size for array 'jeta': expected (2,ncol,nlay,nflav).");
        if (jpress.size()          != ncol * nlay)                     throw std::runtime_error("Invalid size for array 'jpress': expected (ncol, nlay).");

        py::buffer_info buf_flavor = flavor.request();
        py::buffer_info buf_press_ref_log = press_ref_log.request();
        py::buffer_info buf_temp_ref = temp_ref.request();
        py::buffer_info buf_vmr_ref = vmr_ref.request();
        py::buffer_info buf_play = play.request();
        py::buffer_info buf_tlay = tlay.request();
        py::buffer_info buf_col_gas = col_gas.request();
        py::buffer_info buf_jtemp = jtemp.request();
        py::buffer_info buf_fmajor = fmajor.request();
        py::buffer_info buf_fminor = fminor.request();
        py::buffer_info buf_col_mix = col_mix.request();
        py::buffer_info buf_tropo = tropo.request();
        py::buffer_info buf_jeta = jeta.request();
        py::buffer_info buf_jpress = jpress.request();

        fortran::rrtmgp_interpolation(
            ncol,
            nlay,
            ngas,
            nflav,
            neta,
            npres,
            ntemp,
            reinterpret_cast<int *>(buf_flavor.ptr),
            reinterpret_cast<Float *>(buf_press_ref_log.ptr),
            reinterpret_cast<Float *>(buf_temp_ref.ptr),
            press_ref_log_delta,
            temp_ref_min,
            temp_ref_delta,
            press_ref_trop_log,
            reinterpret_cast<Float *>(buf_vmr_ref.ptr),
            reinterpret_cast<Float *>(buf_play.ptr),
            reinterpret_cast<Float *>(buf_tlay.ptr),
            reinterpret_cast<Float *>(buf_col_gas.ptr),
            reinterpret_cast<int *>(buf_jtemp.ptr),
            reinterpret_cast<Float *>(buf_fmajor.ptr),
            reinterpret_cast<Float *>(buf_fminor.ptr),
            reinterpret_cast<Float *>(buf_col_mix.ptr),
            reinterpret_cast<int *>(buf_tropo.ptr),
            reinterpret_cast<int *>(buf_jeta.ptr),
            reinterpret_cast<int *>(buf_jpress.ptr)
        );
    });


    m.def("rrtmgp_compute_tau_absorption",
    [](
        int ncol,
        int nlay,
        int nband,
        int ngpt,
        int ngas,
        int nflav,
        int neta,
        int npres,
        int ntemp,
        int nminorlower,
        int nminorklower,
        int nminorupper,
        int nminorkupper,
        int idx_h2o,
        py::array_t<int> gpoint_flavor,
        py::array_t<int> band_lims_gpt,
        py::array_t<Float> kmajor,
        py::array_t<Float> kminor_lower,
        py::array_t<Float> kminor_upper,
        py::array_t<int> minor_limits_gpt_lower,
        py::array_t<int> minor_limits_gpt_upper,
        py::array_t<Bool> minor_scales_with_density_lower,
        py::array_t<Bool> minor_scales_with_density_upper,
        py::array_t<Bool> scale_by_complement_lower,
        py::array_t<Bool> scale_by_complement_upper,
        py::array_t<int> idx_minor_lower,
        py::array_t<int> idx_minor_upper,
        py::array_t<int> idx_minor_scaling_lower,
        py::array_t<int> idx_minor_scaling_upper,
        py::array_t<int> kminor_start_lower,
        py::array_t<int> kminor_start_upper,
        py::array_t<Bool> tropo,
        py::array_t<Float> col_mix,
        py::array_t<Float> fmajor,
        py::array_t<Float> fminor,
        py::array_t<Float> play,
        py::array_t<Float> tlay,
        py::array_t<Float> col_gas,
        py::array_t<int> jeta,
        py::array_t<int> jtemp,
        py::array_t<int> jpress,
        py::array_t<Float> tau
    ) {
        if (ncol <= 0 || nlay <= 0 || nband <= 0 || ngpt <= 0 || ngas <= 0 || nflav <= 0 || neta <= 0 || 
            npres <= 0 || ntemp <= 0 || nminorlower < 0 || nminorklower < 0 || nminorupper < 0 || nminorkupper < 0) {
            throw std::runtime_error("All input dimensions must be positive integers or zero when appropriate.");
        }

        if (gpoint_flavor.size()               != 2 * ngpt)                          throw std::runtime_error("Invalid size for 'gpoint_flavor': expected (2, ngpt).");
        if (band_lims_gpt.size()               != 2 * nband)                         throw std::runtime_error("Invalid size for 'band_lims_gpt': expected (2, nband).");
        if (kmajor.size()                      != ntemp * neta * (npres + 1) * ngpt) throw std::runtime_error("Invalid size for 'kmajor': expected (ntemp, neta, npres+1, ngpt).");
        if (kminor_lower.size()                != ntemp * neta * nminorklower)       throw std::runtime_error("Invalid size for 'kminor_lower': expected (ntemp, neta, nminorklower).");
        if (kminor_upper.size()                != ntemp * neta * nminorkupper)       throw std::runtime_error("Invalid size for 'kminor_upper': expected (ntemp, neta, nminorkupper).");
        if (minor_limits_gpt_lower.size()      != 2 * nminorlower)                   throw std::runtime_error("Invalid size for 'minor_limits_gpt_lower': expected (2, nminorlower).");
        if (minor_limits_gpt_upper.size()      != 2 * nminorupper)                   throw std::runtime_error("Invalid size for 'minor_limits_gpt_upper': expected (2, nminorupper).");
        if (minor_scales_with_density_lower.size() != nminorlower)                   throw std::runtime_error("Invalid size for 'minor_scales_with_density_lower': expected (nminorlower).");
        if (minor_scales_with_density_upper.size() != nminorupper)                   throw std::runtime_error("Invalid size for 'minor_scales_with_density_upper': expected (nminorupper).");
        if (scale_by_complement_lower.size()   != nminorlower)                       throw std::runtime_error("Invalid size for 'scale_by_complement_lower': expected (nminorlower).");
        if (scale_by_complement_upper.size()   != nminorupper)                       throw std::runtime_error("Invalid size for 'scale_by_complement_upper': expected (nminorupper).");
        if (idx_minor_lower.size()             != nminorlower)                       throw std::runtime_error("Invalid size for 'idx_minor_lower': expected (nminorlower).");
        if (idx_minor_upper.size()             != nminorupper)                       throw std::runtime_error("Invalid size for 'idx_minor_upper': expected (nminorupper).");
        if (idx_minor_scaling_lower.size()     != nminorlower)                       throw std::runtime_error("Invalid size for 'idx_minor_scaling_lower': expected (nminorlower).");
        if (idx_minor_scaling_upper.size()     != nminorupper)                       throw std::runtime_error("Invalid size for 'idx_minor_scaling_upper': expected (nminorupper).");
        if (kminor_start_lower.size()          != nminorlower)                       throw std::runtime_error("Invalid size for 'kminor_start_lower': expected (nminorlower).");
        if (kminor_start_upper.size()          != nminorupper)                       throw std::runtime_error("Invalid size for 'kminor_start_upper': expected (nminorupper).");
        if (tropo.size()                       != ncol * nlay)                       throw std::runtime_error("Invalid size for 'tropo': expected (ncol, nlay).");
        if (col_mix.size()                     != 2 * ncol * nlay * nflav)           throw std::runtime_error("Invalid size for 'col_mix': expected (2, ncol, nlay, nflav).");
        if (fmajor.size()                      != 2 * 2 * 2 * ncol * nlay * nflav)   throw std::runtime_error("Invalid size for 'fmajor': expected (2, 2, 2, ncol, nlay, nflav).");
        if (fminor.size()                      != 2 * 2 * ncol * nlay * nflav)       throw std::runtime_error("Invalid size for 'fminor': expected (2, 2, ncol, nlay, nflav).");
        if (play.size()                        != ncol * nlay)                       throw std::runtime_error("Invalid size for 'play': expected (ncol, nlay).");
        if (tlay.size()                        != ncol * nlay)                       throw std::runtime_error("Invalid size for 'tlay': expected (ncol, nlay).");
        if (col_gas.size()                     != ncol * nlay * (ngas + 1))          throw std::runtime_error("Invalid size for 'col_gas': expected (ncol, nlay, ngas+1).");
        if (jeta.size()                        != 2 * ncol * nlay * nflav)           throw std::runtime_error("Invalid size for 'jeta': expected (2, ncol, nlay, nflav).");
        if (jtemp.size()                       != ncol * nlay)                       throw std::runtime_error("Invalid size for 'jtemp': expected (ncol, nlay).");
        if (jpress.size()                      != ncol * nlay)                       throw std::runtime_error("Invalid size for 'jpress': expected (ncol, nlay).");
        if (tau.size()                         != ncol * nlay * ngpt)                throw std::runtime_error("Invalid size for 'tau': expected (ncol, nlay, ngpt).");

        py::buffer_info buf_gpoint_flavor = gpoint_flavor.request();
        py::buffer_info buf_band_lims_gpt = band_lims_gpt.request();
        py::buffer_info buf_kmajor = kmajor.request();
        py::buffer_info buf_kminor_lower = kminor_lower.request();
        py::buffer_info buf_kminor_upper = kminor_upper.request();
        py::buffer_info buf_minor_limits_gpt_lower = minor_limits_gpt_lower.request();
        py::buffer_info buf_minor_limits_gpt_upper = minor_limits_gpt_upper.request();
        py::buffer_info buf_minor_scales_with_density_lower = minor_scales_with_density_lower.request();
        py::buffer_info buf_minor_scales_with_density_upper = minor_scales_with_density_upper.request();
        py::buffer_info buf_scale_by_complement_lower = scale_by_complement_lower.request();
        py::buffer_info buf_scale_by_complement_upper = scale_by_complement_upper.request();
        py::buffer_info buf_idx_minor_lower = idx_minor_lower.request();
        py::buffer_info buf_idx_minor_upper = idx_minor_upper.request();
        py::buffer_info buf_idx_minor_scaling_lower = idx_minor_scaling_lower.request();
        py::buffer_info buf_idx_minor_scaling_upper = idx_minor_scaling_upper.request();
        py::buffer_info buf_kminor_start_lower = kminor_start_lower.request();
        py::buffer_info buf_kminor_start_upper = kminor_start_upper.request();
        py::buffer_info buf_tropo = tropo.request();
        py::buffer_info buf_col_mix = col_mix.request();
        py::buffer_info buf_fmajor = fmajor.request();
        py::buffer_info buf_fminor = fminor.request();
        py::buffer_info buf_play = play.request();
        py::buffer_info buf_tlay = tlay.request();
        py::buffer_info buf_col_gas = col_gas.request();
        py::buffer_info buf_jeta = jeta.request();
        py::buffer_info buf_jtemp = jtemp.request();
        py::buffer_info buf_jpress = jpress.request();
        py::buffer_info buf_tau = tau.request();

        fortran::rrtmgp_compute_tau_absorption(
            ncol,
            nlay,
            nband,
            ngpt,
            ngas,
            nflav,
            neta,
            npres,
            ntemp,
            nminorlower,
            nminorklower,
            nminorupper,
            nminorkupper,
            idx_h2o,
            reinterpret_cast<int*>(buf_gpoint_flavor.ptr),
            reinterpret_cast<int*>(buf_band_lims_gpt.ptr),
            reinterpret_cast<Float*>(buf_kmajor.ptr),
            reinterpret_cast<Float*>(buf_kminor_lower.ptr),
            reinterpret_cast<Float*>(buf_kminor_upper.ptr),
            reinterpret_cast<int*>(buf_minor_limits_gpt_lower.ptr),
            reinterpret_cast<int*>(buf_minor_limits_gpt_upper.ptr),
            reinterpret_cast<int*>(buf_minor_scales_with_density_lower.ptr),
            reinterpret_cast<int*>(buf_minor_scales_with_density_upper.ptr),
            reinterpret_cast<int*>(buf_scale_by_complement_lower.ptr),
            reinterpret_cast<int*>(buf_scale_by_complement_upper.ptr),
            reinterpret_cast<int*>(buf_idx_minor_lower.ptr),
            reinterpret_cast<int*>(buf_idx_minor_upper.ptr),
            reinterpret_cast<int*>(buf_idx_minor_scaling_lower.ptr),
            reinterpret_cast<int*>(buf_idx_minor_scaling_upper.ptr),
            reinterpret_cast<int*>(buf_kminor_start_lower.ptr),
            reinterpret_cast<int*>(buf_kminor_start_upper.ptr),
            reinterpret_cast<int*>(buf_tropo.ptr),
            reinterpret_cast<Float*>(buf_col_mix.ptr),
            reinterpret_cast<Float*>(buf_fmajor.ptr),
            reinterpret_cast<Float*>(buf_fminor.ptr),
            reinterpret_cast<Float*>(buf_play.ptr),
            reinterpret_cast<Float*>(buf_tlay.ptr),
            reinterpret_cast<Float*>(buf_col_gas.ptr),
            reinterpret_cast<int*>(buf_jeta.ptr),
            reinterpret_cast<int*>(buf_jtemp.ptr),
            reinterpret_cast<int*>(buf_jpress.ptr),
            reinterpret_cast<Float*>(buf_tau.ptr)
        );
    });


    m.def("rrtmgp_compute_Planck_source",
    [](
        int ncol,
        int nlay,
        int nbnd,
        int ngpt,
        int nflav,
        int neta,
        int npres,
        int ntemp,
        int nPlanckTemp,
        py::array_t<Float> tlay,
        py::array_t<Float> tlev,
        py::array_t<Float> tsfc,
        int sfc_lay,
        py::array_t<Float> fmajor,
        py::array_t<int> jeta,
        py::array_t<Bool> tropo,
        py::array_t<int> jtemp,
        py::array_t<int> jpress,
        py::array_t<int> gpoint_bands,
        py::array_t<int> band_lims_gpt,
        py::array_t<Float> pfracin,
        Float temp_ref_min,
        Float totplnk_delta,
        py::array_t<Float> totplnk,
        py::array_t<int> gpoint_flavor,
        py::array_t<Float> sfc_src,
        py::array_t<Float> lay_src,
        py::array_t<Float> lev_src,
        py::array_t<Float> sfc_src_jac
    ) {
        if (ncol <= 0 || nlay <= 0 || nbnd <= 0 || ngpt <= 0 || nflav <= 0 || neta <= 0 || npres <= 0 || ntemp <= 0 || nPlanckTemp <= 0) {
            throw std::runtime_error("All input dimensions must be positive integers");
        }

        if (tlay.size()            != ncol * nlay)                       throw std::runtime_error("Invalid size for 'tlay': expected (ncol, nlay).");
        if (tlev.size()            != ncol * (nlay + 1))                 throw std::runtime_error("Invalid size for 'tlev': expected (ncol, nlay+1).");
        if (tsfc.size()            != ncol)                              throw std::runtime_error("Invalid size for 'tsfc': expected (ncol).");
        if (fmajor.size()          != 2 * 2 * 2 * ncol * nlay * nflav)   throw std::runtime_error("Invalid size for 'fmajor': expected (2, 2, 2, ncol, nlay, nflav).");
        if (jeta.size()            != 2 * ncol * nlay * nflav)           throw std::runtime_error("Invalid size for 'jeta': expected (2, ncol, nlay, nflav).");
        if (tropo.size()           != ncol * nlay)                       throw std::runtime_error("Invalid size for 'tropo': expected (ncol, nlay).");
        if (jtemp.size()           != ncol * nlay)                       throw std::runtime_error("Invalid size for 'jtemp': expected (ncol, nlay).");
        if (jpress.size()          != ncol * nlay)                       throw std::runtime_error("Invalid size for 'jpress': expected (ncol, nlay).");
        if (gpoint_bands.size()    != ngpt)                              throw std::runtime_error("Invalid size for 'gpoint_bands': expected (ngpt).");
        if (band_lims_gpt.size()   != 2 * nbnd)                          throw std::runtime_error("Invalid size for 'band_lims_gpt': expected (2, nbnd).");
        if (pfracin.size()         != ntemp * neta * (npres + 1) * ngpt) throw std::runtime_error("Invalid size for 'pfracin': expected (ntemp, neta, npres+1, ngpt).");
        if (totplnk.size()         != nPlanckTemp * nbnd)                throw std::runtime_error("Invalid size for 'totplnk': expected (nPlanckTemp, nbnd).");
        if (gpoint_flavor.size()   != 2 * ngpt)                          throw std::runtime_error("Invalid size for 'gpoint_flavor': expected (2, ngpt).");
        if (sfc_src.size()         != ncol * ngpt)                       throw std::runtime_error("Invalid size for 'sfc_src': expected (ncol, ngpt).");
        if (lay_src.size()         != ncol * nlay * ngpt)                throw std::runtime_error("Invalid size for 'lay_src': expected (ncol, nlay, ngpt).");
        if (lev_src.size()         != ncol * (nlay + 1) * ngpt)          throw std::runtime_error("Invalid size for 'lev_src': expected (ncol, nlay+1, ngpt).");
        if (sfc_src_jac.size()     != ncol * ngpt)                       throw std::runtime_error("Invalid size for 'sfc_src_jac': expected (ncol, ngpt).");

        py::buffer_info buf_tlay = tlay.request();
        py::buffer_info buf_tlev = tlev.request();
        py::buffer_info buf_tsfc = tsfc.request();
        py::buffer_info buf_fmajor = fmajor.request();
        py::buffer_info buf_jeta = jeta.request();
        py::buffer_info buf_tropo = tropo.request();
        py::buffer_info buf_jtemp = jtemp.request();
        py::buffer_info buf_jpress = jpress.request();
        py::buffer_info buf_gpoint_bands = gpoint_bands.request();
        py::buffer_info buf_band_lims_gpt = band_lims_gpt.request();
        py::buffer_info buf_pfracin = pfracin.request();
        py::buffer_info buf_totplnk = totplnk.request();
        py::buffer_info buf_gpoint_flavor = gpoint_flavor.request();
        py::buffer_info buf_sfc_src = sfc_src.request();
        py::buffer_info buf_lay_src = lay_src.request();
        py::buffer_info buf_lev_src = lev_src.request();
        py::buffer_info buf_sfc_src_jac = sfc_src_jac.request();

        fortran::rrtmgp_compute_Planck_source(
            ncol,
            nlay,
            nbnd,
            ngpt,
            nflav,
            neta,
            npres,
            ntemp,
            nPlanckTemp,
            reinterpret_cast<Float*>(buf_tlay.ptr),
            reinterpret_cast<Float*>(buf_tlev.ptr),
            reinterpret_cast<Float*>(buf_tsfc.ptr),
            sfc_lay,
            reinterpret_cast<Float*>(buf_fmajor.ptr),
            reinterpret_cast<int*>(buf_jeta.ptr),
            reinterpret_cast<int*>(buf_tropo.ptr),
            reinterpret_cast<int*>(buf_jtemp.ptr),
            reinterpret_cast<int*>(buf_jpress.ptr),
            reinterpret_cast<int*>(buf_gpoint_bands.ptr),
            reinterpret_cast<int*>(buf_band_lims_gpt.ptr),
            reinterpret_cast<Float*>(buf_pfracin.ptr),
            temp_ref_min,
            totplnk_delta,
            reinterpret_cast<Float*>(buf_totplnk.ptr),
            reinterpret_cast<int*>(buf_gpoint_flavor.ptr),
            reinterpret_cast<Float*>(buf_sfc_src.ptr),
            reinterpret_cast<Float*>(buf_lay_src.ptr),
            reinterpret_cast<Float*>(buf_lev_src.ptr),
            reinterpret_cast<Float*>(buf_sfc_src_jac.ptr)
        );
    });

    m.def("rrtmgp_compute_tau_rayleigh",
    [](
        int ncol,
        int nlay,
        int nbnd,
        int ngpt,
        int ngas,
        int nflav,
        int neta,
        int npres,
        int ntemp,
        py::array_t<int> gpoint_flavor,
        py::array_t<int> band_lims_gpt,
        py::array_t<Float> krayl,
        int idx_h2o,
        py::array_t<Float> col_dry,
        py::array_t<Float> col_gas,
        py::array_t<Float> fminor,
        py::array_t<int> jeta,
        py::array_t<Bool> tropo,
        py::array_t<int> jtemp,
        py::array_t<Float> tau_rayleigh
    ) {
        if (ncol <= 0 || nlay  <= 0 || nbnd  <= 0 ||
            ngpt <= 0 || ngas  <= 0 || nflav <= 0 ||
            neta <= 0 || ntemp <= 0
        ) {
            throw std::runtime_error("ncol, nlay, nbnd, ngpt, ngas, nflav, neta and ntemp must be positive integers");
        }

        if (gpoint_flavor.size() != 2 * ngpt)                    throw std::runtime_error("Invalid size for input array 'gpoint_flavor'");
        if (band_lims_gpt.size() != 2 * nbnd)                    throw std::runtime_error("Invalid size for input array 'band_lims_gpt'");
        if (krayl.size()         != ntemp * neta * ngpt * 2)     throw std::runtime_error("Invalid size for input array 'krayl'");
        if (col_dry.size()       != ncol * nlay)                 throw std::runtime_error("Invalid size for input array 'col_dry'");
        if (col_gas.size()       != ncol * nlay * (ngas + 1))    throw std::runtime_error("Invalid size for input array 'col_gas'");
        if (fminor.size()        != 2 * 2 * ncol * nlay * nflav) throw std::runtime_error("Invalid size for input array 'fminor'");
        if (jeta.size()          != 2 * ncol * nlay * nflav)     throw std::runtime_error("Invalid size for input array 'jeta'");
        if (tropo.size()         != ncol * nlay)                 throw std::runtime_error("Invalid size for input array 'tropo'");
        if (jtemp.size()         != ncol * nlay)                 throw std::runtime_error("Invalid size for input array 'jtemp'");
        if (tau_rayleigh.size()  != ncol * nlay * ngpt)          throw std::runtime_error("Invalid size for input array 'tau_rayleigh'");

        py::buffer_info buf_gpoint_flavor = gpoint_flavor.request();
        py::buffer_info buf_band_lims_gpt = band_lims_gpt.request();
        py::buffer_info buf_krayl         = krayl.request();
        py::buffer_info buf_col_dry       = col_dry.request();
        py::buffer_info buf_col_gas       = col_gas.request();
        py::buffer_info buf_fminor        = fminor.request();
        py::buffer_info buf_jeta          = jeta.request();
        py::buffer_info buf_tropo         = tropo.request();
        py::buffer_info buf_jtemp         = jtemp.request();
        py::buffer_info buf_tau_rayleigh  = tau_rayleigh.request();

        fortran::rrtmgp_compute_tau_rayleigh(
            ncol,
            nlay,
            nbnd,
            ngpt,
            ngas,
            nflav,
            neta,
            npres,
            ntemp,
            reinterpret_cast<int*> (buf_gpoint_flavor.ptr),
            reinterpret_cast<int*> (buf_band_lims_gpt.ptr),
            reinterpret_cast<Float*> (buf_krayl.ptr),
            idx_h2o,
            reinterpret_cast<Float*> (buf_col_dry.ptr),
            reinterpret_cast<Float*> (buf_col_gas.ptr),
            reinterpret_cast<Float*> (buf_fminor.ptr),
            reinterpret_cast<int*> (buf_jeta.ptr),
            reinterpret_cast<int*> (buf_tropo.ptr),
            reinterpret_cast<int*> (buf_jtemp.ptr),
            reinterpret_cast<Float*> (buf_tau_rayleigh.ptr)
        );
    });

    m.def("rrtmgp_compute_cld_from_table",
    [](
        int ncol,
        int nlay,
        int nbnd,
        py::array_t<Bool>  mask,
        py::array_t<Float> lwp,
        py::array_t<Float> re,
        int nsteps,
        Float step_size,
        Float offset,
        py::array_t<Float> tau_table,
        py::array_t<Float> ssa_table,
        py::array_t<Float> asy_table,
        py::array_t<Float> tau,
        py::array_t<Float> taussa,
        py::array_t<Float> taussag
    ) {
        if (ncol <= 0 || nlay <= 0 || nbnd <= 0 || nsteps <= 0) {
            throw std::runtime_error("ncol, nlay, nbnd and nsteps must be positive integers");
        }

        if (mask.size()      != ncol * nlay)        throw std::runtime_error("Invalid size for input array 'mask'");
        if (lwp.size()       != ncol * nlay)        throw std::runtime_error("Invalid size for input array 'lwp'");
        if (re.size()        != ncol * nlay)        throw std::runtime_error("Invalid size for input array 're'");
        if (tau_table.size() != nsteps * nbnd)      throw std::runtime_error("Invalid size for input array 'tau_table'");
        if (ssa_table.size() != nsteps * nbnd)      throw std::runtime_error("Invalid size for input array 'ssa_table'");
        if (asy_table.size() != nsteps * nbnd)      throw std::runtime_error("Invalid size for input array 'asy_table'");
        if (tau.size()       != ncol * nlay * nbnd) throw std::runtime_error("Invalid size for input array 'tau'");
        if (taussa.size()    != ncol * nlay * nbnd) throw std::runtime_error("Invalid size for input array 'taussa'");
        if (taussag.size()   != ncol * nlay * nbnd) throw std::runtime_error("Invalid size for input array 'taussag'");

        py::buffer_info buf_mask      = mask.request();
        py::buffer_info buf_lwp       = lwp.request();
        py::buffer_info buf_re        = re.request();
        py::buffer_info buf_tau_table = tau_table.request();
        py::buffer_info buf_ssa_table = ssa_table.request();
        py::buffer_info buf_asy_table = asy_table.request();
        py::buffer_info buf_tau       = tau.request();
        py::buffer_info buf_taussa    = taussa.request();
        py::buffer_info buf_taussag   = taussag.request();

        fortran::rrtmgp_compute_cld_from_table(
            ncol,
            nlay,
            nbnd,
            reinterpret_cast<Bool*>(buf_mask.ptr),
            reinterpret_cast<Float*>(buf_lwp.ptr),
            reinterpret_cast<Float*>(buf_re.ptr),
            nsteps,
            step_size,
            offset,
            reinterpret_cast<Float*>(buf_tau_table.ptr),
            reinterpret_cast<Float*>(buf_ssa_table.ptr),
            reinterpret_cast<Float*>(buf_asy_table.ptr),
            reinterpret_cast<Float*>(buf_tau.ptr),
            reinterpret_cast<Float*>(buf_taussa.ptr),
            reinterpret_cast<Float*>(buf_taussag.ptr)
        );
    });
}
