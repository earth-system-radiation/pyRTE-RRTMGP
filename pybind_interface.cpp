#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace fortran {
#include "rte_kernels.h"
}

namespace py = pybind11;

PYBIND11_MODULE(pyrte, m) {

    m.def("rte_lw_solver_noscat",
    [](
        int ncol, int nlay, int ngpt, bool top_at_1, int nmus,
        py::array_t<double> Ds, py::array_t<double> weights,
        py::array_t<double> tau, py::array_t<double> lay_source,
        py::array_t<double> lev_source, py::array_t<double> sfc_emis,
        py::array_t<double> sfc_src, py::array_t<double> inc_flux,
        py::array_t<double> flux_up, py::array_t<double> flux_dn,
        bool do_broadband, py::array_t<double> broadband_up,
        py::array_t<double> broadband_dn, bool do_Jacobians,
        py::array_t<double> sfc_srcJac, py::array_t<double> flux_upJac,
        bool do_rescaling, py::array_t<double> ssa, py::array_t<double> g
    ) {
        int top_at_1_int = int(top_at_1);
        int do_broadband_int = int(do_broadband);
        int do_Jacobians_int = int(do_Jacobians);
        int do_rescaling_int = int(do_rescaling);

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
            &ncol,
            &nlay,
            &ngpt,
            &top_at_1_int,
            &nmus,
            reinterpret_cast<double *>(buf_Ds.ptr),
            reinterpret_cast<double *>(buf_weights.ptr),
            reinterpret_cast<double *>(buf_tau.ptr),
            reinterpret_cast<double *>(buf_lay_source.ptr),
            reinterpret_cast<double *>(buf_lev_source.ptr),
            reinterpret_cast<double *>(buf_sfc_emis.ptr),
            reinterpret_cast<double *>(buf_sfc_src.ptr),
            reinterpret_cast<double *>(buf_inc_flux.ptr),
            reinterpret_cast<double *>(buf_flux_up.ptr),
            reinterpret_cast<double *>(buf_flux_dn.ptr),
            &do_broadband_int,
            reinterpret_cast<double *>(buf_broadband_up.ptr),
            reinterpret_cast<double *>(buf_broadband_dn.ptr),
            &do_Jacobians_int,
            reinterpret_cast<double *>(buf_sfc_srcJac.ptr),
            reinterpret_cast<double *>(buf_flux_upJac.ptr),
            &do_rescaling_int,
            reinterpret_cast<double *>(buf_ssa.ptr),
            reinterpret_cast<double *>(buf_g.ptr)
        );
    });

    m.def("rte_sw_solver_noscat",
    [](
        int ncol, int nlay, int ngpt, bool top_at_1,
        py::array_t<double> tau, py::array_t<double> mu0,
        py::array_t<double> inc_flux_dir, py::array_t<double> flux_dir
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (tau.size() != ncol * nlay * ngpt || mu0.size() != ncol * nlay || inc_flux_dir.size() != ncol * ngpt) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        if (flux_dir.ndim() != 3 || flux_dir.shape(0) != ncol || flux_dir.shape(1) != nlay + 1 || flux_dir.shape(2) != ngpt) {
            throw std::runtime_error("Invalid dimensions for flux_dir array");
        }

        int top_at_1_int = int(top_at_1);

        py::buffer_info buf_tau = tau.request();
        py::buffer_info buf_mu0 = mu0.request();
        py::buffer_info buf_inc_flux_dir = inc_flux_dir.request();
        py::buffer_info buf_flux_dir = flux_dir.request();

        fortran::rte_sw_solver_noscat(
            &ncol,
            &nlay,
            &ngpt,
            &top_at_1_int,
            reinterpret_cast<double *>(buf_tau.ptr),
            reinterpret_cast<double *>(buf_mu0.ptr),
            reinterpret_cast<double *>(buf_inc_flux_dir.ptr),
            reinterpret_cast<double *>(buf_flux_dir.ptr)
        );
    });


    m.def("rte_sw_solver_2stream",
    [](
        int ncol, int nlay, int ngpt, bool top_at_1,
        py::array_t<double> tau, py::array_t<double> ssa, py::array_t<double> g,
        py::array_t<double> mu0, py::array_t<double> sfc_alb_dir,
        py::array_t<double> sfc_alb_dif, py::array_t<double> inc_flux_dir,
        py::array_t<double> flux_up, py::array_t<double> flux_dn,
        py::array_t<double> flux_dir, bool has_dif_bc, py::array_t<double> inc_flux_dif,
        bool do_broadband, py::array_t<double> broadband_up, py::array_t<double> broadband_dn,
        py::array_t<double> broadband_dir
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            tau.size() != ncol * nlay * ngpt ||
            ssa.size() != ncol * nlay * ngpt ||
            g.size() != ncol * nlay * ngpt ||
            mu0.size() != ncol * nlay ||
            sfc_alb_dir.size() != ncol * ngpt ||
            sfc_alb_dif.size() != ncol * ngpt ||
            inc_flux_dir.size() != ncol * ngpt ||
            flux_up.size() != ncol * (nlay + 1) * ngpt ||
            flux_dn.size() != ncol * (nlay + 1) * ngpt ||
            flux_dir.size() != ncol * (nlay + 1) * ngpt ||
            inc_flux_dif.size() != ncol * ngpt ||
            broadband_up.size() != ncol * (nlay + 1) ||
            broadband_dn.size() != ncol * (nlay + 1) ||
            broadband_dir.size() != ncol * (nlay + 1)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        int top_at_1_int = int(top_at_1);
        int has_dif_bc_int = int(has_dif_bc);
        int do_broadband_int = int(do_broadband);

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
            &ncol,
            &nlay,
            &ngpt,
            &top_at_1_int,
            reinterpret_cast<double *>(buf_tau.ptr),
            reinterpret_cast<double *>(buf_ssa.ptr),
            reinterpret_cast<double *>(buf_g.ptr),
            reinterpret_cast<double *>(buf_mu0.ptr),
            reinterpret_cast<double *>(buf_sfc_alb_dir.ptr),
            reinterpret_cast<double *>(buf_sfc_alb_dif.ptr),
            reinterpret_cast<double *>(buf_inc_flux_dir.ptr),
            reinterpret_cast<double *>(buf_flux_up.ptr),
            reinterpret_cast<double *>(buf_flux_dn.ptr),
            reinterpret_cast<double *>(buf_flux_dir.ptr),
            &has_dif_bc_int,
            reinterpret_cast<double *>(buf_inc_flux_dif.ptr),
            &do_broadband_int,
            reinterpret_cast<double *>(buf_broadband_up.ptr),
            reinterpret_cast<double *>(buf_broadband_dn.ptr),
            reinterpret_cast<double *>(buf_broadband_dir.ptr)
        );
    });

    m.def("rte_lw_solver_2stream",
    [](
        int ncol, int nlay, int ngpt, bool top_at_1,
        py::array_t<double> tau, py::array_t<double> ssa, py::array_t<double> g,
        py::array_t<double> lay_source, py::array_t<double> lev_source,
        py::array_t<double> sfc_emis, py::array_t<double> sfc_src,
        py::array_t<double> inc_flux, py::array_t<double> flux_up,
        py::array_t<double> flux_dn
    ) {
        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        if (
            tau.size() != ncol * nlay * ngpt ||
            ssa.size() != ncol * nlay * ngpt ||
            g.size()   != ncol * nlay * ngpt ||
            lay_source.size() != ncol * nlay * ngpt ||
            lev_source.size() != ncol * (nlay + 1) * ngpt ||
            sfc_emis.size() != ncol * ngpt ||
            sfc_src.size()  != ncol * ngpt ||
            inc_flux.size() != ncol * ngpt ||
            flux_up.size()  != ncol * (nlay + 1) * ngpt ||
            flux_dn.size()  != ncol * (nlay + 1) * ngpt
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        int top_at_1_int = int(top_at_1);

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
            &ncol,
            &nlay,
            &ngpt,
            &top_at_1_int,
            reinterpret_cast<double *>(buf_tau.ptr),
            reinterpret_cast<double *>(buf_ssa.ptr),
            reinterpret_cast<double *>(buf_g.ptr),
            reinterpret_cast<double *>(buf_lay_source.ptr),
            reinterpret_cast<double *>(buf_lev_source.ptr),
            reinterpret_cast<double *>(buf_sfc_emis.ptr),
            reinterpret_cast<double *>(buf_sfc_src.ptr),
            reinterpret_cast<double *>(buf_inc_flux.ptr),
            reinterpret_cast<double *>(buf_flux_up.ptr),
            reinterpret_cast<double *>(buf_flux_dn.ptr)
        );
    });

    m.def("rte_increment_1scalar_by_1scalar",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<double> tau_inout,
        py::array_t<double> tau_in
    ) {
        if (tau_inout.size() != ncol * nlay * ngpt ||
            tau_in.size() != ncol * nlay * ngpt
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();

        fortran::rte_increment_1scalar_by_1scalar(&ncol, &nlay, &ngpt, reinterpret_cast<double *>(buf_tau_inout.ptr), reinterpret_cast<double *>(buf_tau_in.ptr));
    });

    m.def("rte_increment_1scalar_by_2stream",
    [](
        int ncol,
        int nlay,
        int ngpt,
        py::array_t<double> tau_inout,
        py::array_t<double> tau_in,
        py::array_t<double> ssa_in
    ) {
        if (
            (tau_inout.size() != ncol * nlay * ngpt) ||
            (tau_in.size() != ncol * nlay * ngpt) ||
            (ssa_in.size() != ncol * nlay * ngpt)
        ) {
            throw std::runtime_error("Invalid size for input arrays");
        }

        if (ncol <= 0 || nlay <= 0 || ngpt <= 0) {
            throw std::runtime_error("ncol, nlay, and ngpt must be positive integers");
        }

        py::buffer_info buf_tau_inout = tau_inout.request();
        py::buffer_info buf_tau_in = tau_in.request();
        py::buffer_info buf_ssa_in = ssa_in.request();

        fortran::rte_increment_1scalar_by_2stream(
            &ncol,
            &nlay,
            &ngpt,
            reinterpret_cast<double *>(buf_tau_inout.ptr),
            reinterpret_cast<double *>(buf_tau_in.ptr),
            reinterpret_cast<double *>(buf_ssa_in.ptr));
    });

    m.def("zero_array_1D", [](py::array_t<double> arr){
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

        int ni = int(buf_info.size);
        double *ptr = reinterpret_cast<double *>(buf_info.ptr);
        fortran::zero_array_1D(&ni, ptr);
    });


    m.def("zero_array_2D", [](py::array_t<double> arr){
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

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);

        double *ptr = reinterpret_cast<double *>(buf_info.ptr);
        fortran::zero_array_2D(&ni, &nj, ptr);
    });

    m.def("zero_array_3D", [](py::array_t<double> arr){
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

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);
        int nk = int(buf_info.shape[2]);

        double *ptr = reinterpret_cast<double *>(buf_info.ptr);
        fortran::zero_array_3D(&ni, &nj, &nk, ptr);
    });

    m.def("zero_array_4D", [](py::array_t<double> arr){
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

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);
        int nk = int(buf_info.shape[2]);
        int nl = int(buf_info.shape[3]);

        double *ptr = reinterpret_cast<double *>(buf_info.ptr);
        fortran::zero_array_4D(&ni, &nj, &nk, &nl, ptr);
    });
}
