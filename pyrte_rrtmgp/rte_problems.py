import numpy as np
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat
from pyrte_rrtmgp.kernels.rte import sw_solver_2stream
from pyrte_rrtmgp.utils import get_usecols, compute_mu0, compute_toa_flux


class LWProblem:
    def __init__(self, tau: np.ndarray, lay_source: np.ndarray, lev_source: np.ndarray,
                 sfc_src: np.ndarray, sfc_src_jac: np.ndarray):
        self.tau = tau
        self.lay_source = lay_source
        self.lev_source = lev_source
        self.sfc_src = sfc_src
        self.sfc_src_jac = sfc_src_jac
        self._sfc_emis = None

    @property
    def sfc_emis(self):
        if self._sfc_emis is None:
            self._sfc_emis = np.ones((self.tau.shape[0], self.tau.shape[-1]))
        return self._sfc_emis

    @sfc_emis.setter
    def sfc_emis(self, value):
        self._sfc_emis = value

    def rte_solve(self):
        """Solve the radiative transfer equation
        
        Returns:
            tuple: Tuple containing (solver_flux_up, solver_flux_down)
        """
        
        _, solver_flux_up, solver_flux_down, _, _ = lw_solver_noscat(
            tau=self.tau,
            lay_source=self.lay_source,
            lev_source=self.lev_source, 
            sfc_emis=self.sfc_emis,
            sfc_src=self.sfc_src,
            sfc_src_jac=self.sfc_src_jac,
        )
        
        return solver_flux_up, solver_flux_down

class SWProblem:
    def __init__(self, tau: np.ndarray, ssa: np.ndarray, g: np.ndarray,
                 sfc_alb_dir: np.ndarray = None,
                 sfc_alb_dif: np.ndarray = None,
                 compute_mu0_fn=compute_mu0,
                 compute_toa_flux_fn=compute_toa_flux,
                 solar_source: np.ndarray = None,
                 solar_zenith_angle: np.ndarray = None,
                 total_solar_irradiance: np.ndarray = None):
        """
        Initialize SW (shortwave) radiative transfer problem.
        """
        self.tau = tau
        self.ssa = ssa
        self.g = g
        self.nlayer = tau.shape[1]
        
        # Store inputs needed for computing mu0 and inc_flux_dir
        self._solar_zenith_angle = solar_zenith_angle
        self._total_solar_irradiance = total_solar_irradiance
        self._solar_source = solar_source
        self._compute_mu0_fn = compute_mu0_fn
        self._compute_toa_flux_fn = compute_toa_flux_fn
        
        # Custom values (initialized as None)
        self._mu0 = None
        self._inc_flux_dir = None
            
        # Surface albedo
        self._sfc_alb_dir = sfc_alb_dir
        self._sfc_alb_dif = sfc_alb_dif

    @property
    def sfc_alb_dir(self):
        """Get direct surface albedo"""
        if self._sfc_alb_dir is None:
            raise ValueError("sfc_alb_dir must be set")
        return self._sfc_alb_dir

    @sfc_alb_dir.setter
    def sfc_alb_dir(self, value):
        """Set direct surface albedo value"""
        self._sfc_alb_dir = value

    @property
    def sfc_alb_dif(self):
        """Get diffuse surface albedo, defaults to direct if not set"""
        if self._sfc_alb_dif is None:
            return self.sfc_alb_dir
        return self._sfc_alb_dif

    @sfc_alb_dif.setter
    def sfc_alb_dif(self, value):
        """Set diffuse surface albedo value"""
        self._sfc_alb_dif = value

    @property
    def solar_zenith_angle(self):
        """Get solar zenith angle"""
        if self._solar_zenith_angle is None:
            raise ValueError("solar_zenith_angle must be set")
        return self._solar_zenith_angle

    @solar_zenith_angle.setter
    def solar_zenith_angle(self, value):
        """Set solar zenith angle value"""
        self._solar_zenith_angle = value

    @property
    def mu0(self):
        """Get mu0 value, computing it from solar_zenith_angle if not set manually"""
        if self._mu0 is not None:
            return self._mu0
        return self._compute_mu0_fn(self.solar_zenith_angle, nlayer=self.nlayer)

    @mu0.setter
    def mu0(self, value):
        """Set custom mu0 value"""
        self._mu0 = value

    @property
    def total_solar_irradiance(self):
        """Get total solar irradiance"""
        if self._total_solar_irradiance is None:
            raise ValueError("total_solar_irradiance must be set")
        return self._total_solar_irradiance

    @total_solar_irradiance.setter
    def total_solar_irradiance(self, value):
        """Set total solar irradiance value"""
        self._total_solar_irradiance = value

    @property
    def inc_flux_dir(self):
        """Get incident flux, computing it from TSI and solar source if not set manually"""
        if self._inc_flux_dir is not None:
            return self._inc_flux_dir
        elif self._solar_source is not None:
            return self._compute_toa_flux_fn(self.total_solar_irradiance, self._solar_source)
        else:
            raise ValueError("Either set inc_flux_dir directly or provide solar_source")

    @inc_flux_dir.setter
    def inc_flux_dir(self, value):
        """Set custom incident flux value"""
        self._inc_flux_dir = value

    def solve(self):
        """Solve the SW radiative transfer problem."""
        # Get mu0 and inc_flux_dir using properties
        mu0 = self.mu0
        inc_flux_dir = self.inc_flux_dir

        _, _, _, flux_up, flux_down, _ = sw_solver_2stream(
            tau=self.tau,
            ssa=self.ssa,
            g=self.g,
            mu0=mu0,
            sfc_alb_dir=self.sfc_alb_dir,
            sfc_alb_dif=self.sfc_alb_dif,
            inc_flux_dir=inc_flux_dir,
        )
        
        # Post-process results for nighttime columns
        if self.solar_zenith_angle is not None:
            usecol = get_usecols(self.solar_zenith_angle)
            flux_up = flux_up * usecol[:, np.newaxis]
            flux_down = flux_down * usecol[:, np.newaxis]

        return flux_up, flux_down