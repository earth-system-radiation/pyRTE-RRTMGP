import os

import numpy as np
import xarray as xr
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
import pandas as pd

rte_rrtmgp_dir = download_rrtmgp_data()



def compute_profiles(SST, ncol, nlay):
    """
    Construct profiles of pressure, temperature, humidity, and ozone
    following the RCEMIP protocol for a surface temperature of 300K.
    Based on Python implementation by Chiel van Heerwardeen.
    """
    # Constants
    z_trop = 15000.0
    z_top = 70.0e3
    g1 = 3.6478
    g2 = 0.83209 
    g3 = 11.3515
    o3_min = 1e-13
    g = 9.79764
    Rd = 287.04
    p0 = 101480.0  # Surface pressure
    z_q1 = 4.0e3
    z_q2 = 7.5e3
    q_t = 1.0e-8
    gamma = 6.7e-3
    q_0 = 0.01864  # for 300 K SST

    # Initialize arrays
    p_lay = np.zeros((ncol, nlay))
    t_lay = np.zeros((ncol, nlay))
    q_lay = np.zeros((ncol, nlay))
    o3 = np.zeros((ncol, nlay))
    p_lev = np.zeros((ncol, nlay+1))
    t_lev = np.zeros((ncol, nlay+1))

    # Initial calculations
    Tv0 = (1.0 + 0.608*q_0) * SST

    # Split resolution above and below RCE tropopause (15 km or about 125 hPa)
    z_lev = np.zeros(nlay+1)
    z_lev[0] = 0.0
    z_lev[1:nlay//2+1] = 2.0 * z_trop/nlay * np.arange(1, nlay//2+1)
    z_lev[nlay//2+1:] = z_trop + 2.0 * (z_top - z_trop)/nlay * np.arange(1, nlay//2+1)
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])

    # Layer calculations
    for ilay in range(nlay):
        for icol in range(ncol):
            z = z_lay[ilay]
            if z > z_trop:
                q = q_t
                T = SST - gamma*z_trop/(1.0 + 0.608*q_0)
                Tv = (1.0 + 0.608*q) * T
                p = p0 * (Tv/Tv0)**(g/(Rd*gamma)) * np.exp(-((g*(z-z_trop))/(Rd*Tv)))
            else:
                q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)
                T = SST - gamma*z / (1.0 + 0.608*q)
                Tv = (1.0 + 0.608*q) * T
                p = p0 * (Tv/Tv0)**(g/(Rd*gamma))

            p_lay[icol,ilay] = p
            t_lay[icol,ilay] = T
            q_lay[icol,ilay] = q
            p_hpa = p_lay[icol,ilay] / 100.0
            o3[icol,ilay] = max(o3_min, g1 * p_hpa**g2 * np.exp(-p_hpa/g3) * 1.0e-6)

    # Level calculations
    for ilay in range(nlay+1):
        for icol in range(ncol):
            z = z_lev[ilay]
            if z > z_trop:
                q = q_t
                T = SST - gamma*z_trop/(1.0 + 0.608*q_0)
                Tv = (1.0 + 0.608*q) * T
                p = p0 * (Tv/Tv0)**(g/(Rd*gamma)) * np.exp(-((g*(z-z_trop))/(Rd*Tv)))
            else:
                q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)
                T = SST - gamma*z / (1.0 + 0.608*q)
                Tv = (1.0 + 0.608*q) * T
                p = p0 * (Tv/Tv0)**(g/(Rd*gamma))

            p_lev[icol,ilay] = p
            t_lev[icol,ilay] = T

    return p_lay, t_lay, p_lev, t_lev, q_lay, o3

def expand_to_2d(value, ncol, nlay, name=None):
    """Expand scalar or 1D array to 2D array with shape (ncol, nlay)"""
    value = np.asarray(value)
    
    if value.ndim == 0:  # Scalar input
        data = np.full((ncol, nlay), value)
        
    elif value.ndim == 1:  # Layer-dependent input
        if len(value) != nlay:
            raise ValueError(f"Layer-dependent value must have length {nlay}")
        data = np.tile(value[np.newaxis, :], (ncol, 1))
        
    elif value.ndim == 2:  # Full 2D specification
        if value.shape != (ncol, nlay):
            raise ValueError(f"2D value must have shape ({ncol}, {nlay})")
        data = value
    else:
        raise ValueError("Invalid dimensions - must be scalar, 1D or 2D array")
        
    return xr.DataArray(
        data,
        dims=["columns", "layers"],
        name=name
    )

def create_gas_dataset(gas_values, ncol, nlay):
    """Create xarray Dataset with gas concentrations.
    
    Args:
        gas_values (dict): Dictionary mapping gas names to concentration values
        ncol (int): Number of columns
        nlay (int): Number of layers
        
    Returns:
        xr.Dataset: Dataset containing gas concentrations as separate variables
    """
    ds = xr.Dataset()
    
    # Convert each gas value to 2D array and add as variable
    for gas_name, value in gas_values.items():
        data_array = expand_to_2d(value, ncol, nlay)
        ds[gas_name] = data_array.rename({"columns": "site", "layers": "layer"})
        
    return ds

def compute_clouds(cloud_optics, ncol, nlay, p_lay, t_lay):
    """
    Compute cloud properties for radiative transfer calculations.
    """
    # Get min/max radii values for liquid and ice
    rel_val = 0.5 * (cloud_optics["radliq_lwr"] + cloud_optics["radliq_upr"])
    rei_val = 0.5 * (cloud_optics["radice_lwr"] + cloud_optics["radice_upr"])

    # Initialize arrays
    cloud_mask = np.zeros((ncol, nlay), dtype=bool)
    lwp = np.zeros((ncol, nlay))  # liquid water path
    iwp = np.zeros((ncol, nlay))  # ice water path
    rel = np.zeros((ncol, nlay))  # effective radius liquid
    rei = np.zeros((ncol, nlay))  # effective radius ice

    # Adjust the modulo operation to match Fortran's 1-based indexing
    for ilay in range(nlay):
        for icol in range(ncol):
            cloud_mask[icol,ilay] = (p_lay[icol,ilay] > 100 * 100 and 
                                   p_lay[icol,ilay] < 900 * 100 and 
                                   (icol + 1) % 3 != 0)  # Add 1 to match Fortran indexing
            
            # Ice and liquid will overlap in a few layers
            if cloud_mask[icol,ilay]:
                if t_lay[icol,ilay] > 263:
                    lwp[icol,ilay] = 10.0
                    rel[icol,ilay] = rel_val
                if t_lay[icol,ilay] < 273:
                    iwp[icol,ilay] = 10.0 
                    rei[icol,ilay] = rei_val

    return lwp, iwp, rel, rei


def compute_all_from_table(ncol, nlay, nbnd, mask, lwp, re, nsteps, step_size, offset, tau_table, ssa_table, asy_table):
    """
    Compute optical properties from lookup tables.
    
    Args:
        ncol (int): Number of columns
        nlay (int): Number of layers 
        nbnd (int): Number of bands
        mask (ndarray): Boolean mask array (ncol, nlay)
        lwp (ndarray): Liquid water path array (ncol, nlay)
        re (ndarray): Effective radius array (ncol, nlay)
        nsteps (int): Number of steps in lookup tables
        step_size (float): Step size for interpolation
        offset (float): Offset for interpolation
        tau_table (ndarray): Optical depth table (nsteps, nbnd)
        ssa_table (ndarray): Single scattering albedo table (nsteps, nbnd)
        asy_table (ndarray): Asymmetry parameter table (nsteps, nbnd)
        
    Returns:
        tuple: Arrays of optical properties (tau, taussa, taussag) each with shape (ncol, nlay, nbnd)
    """
    import numpy as np
    
    # Initialize output arrays
    tau = np.zeros((ncol, nlay, nbnd))
    taussa = np.zeros((ncol, nlay, nbnd))
    taussag = np.zeros((ncol, nlay, nbnd))
    
    for ibnd in range(nbnd):
        for ilay in range(nlay):
            for icol in range(ncol):
                if mask[icol,ilay]:
                    # Calculate interpolation indices and weights
                    index = min(int(np.floor((re[icol,ilay] - offset)/step_size)), nsteps-2)
                    fint = (re[icol,ilay] - offset)/step_size - index
                    
                    # Interpolate optical properties
                    t = lwp[icol,ilay] * (tau_table[index,ibnd] + 
                         fint * (tau_table[index+1,ibnd] - tau_table[index,ibnd]))
                    
                    ts = t * (ssa_table[index,ibnd] + 
                         fint * (ssa_table[index+1,ibnd] - ssa_table[index,ibnd]))
                    
                    taussag[icol,ilay,ibnd] = ts * (asy_table[index,ibnd] + 
                         fint * (asy_table[index+1,ibnd] - asy_table[index,ibnd]))
                         
                    taussa[icol,ilay,ibnd] = ts
                    tau[icol,ilay,ibnd] = t
                    
    return tau, taussa, taussag


def compute_all_from_pade(ncol, nlay, nbnd, nsizes,
                         mask, lwp, re,
                         m_ext, n_ext, re_bounds_ext, coeffs_ext,
                         m_ssa, n_ssa, re_bounds_ssa, coeffs_ssa,
                         m_asy, n_asy, re_bounds_asy, coeffs_asy):
    """Compute optical properties using Pade approximants.
    
    Args:
        ncol (int): Number of columns
        nlay (int): Number of layers
        nbnd (int): Number of bands 
        nsizes (int): Number of size regimes
        mask (ndarray): Boolean mask array (ncol, nlay)
        lwp (ndarray): Liquid water path array (ncol, nlay)
        re (ndarray): Effective radius array (ncol, nlay)
        m_ext, n_ext (int): Orders of Pade approximant for extinction
        re_bounds_ext (ndarray): Size regime boundaries for extinction
        coeffs_ext (ndarray): Pade coefficients for extinction
        m_ssa, n_ssa (int): Orders of Pade approximant for single scattering albedo
        re_bounds_ssa (ndarray): Size regime boundaries for single scattering albedo
        coeffs_ssa (ndarray): Pade coefficients for single scattering albedo
        m_asy, n_asy (int): Orders of Pade approximant for asymmetry parameter
        re_bounds_asy (ndarray): Size regime boundaries for asymmetry parameter
        coeffs_asy (ndarray): Pade coefficients for asymmetry parameter
        
    Returns:
        tuple: Arrays of optical properties (tau, taussa, taussag) each with shape (ncol, nlay, nbnd)
    """
    import numpy as np
    
    # Initialize output arrays
    tau = np.zeros((ncol, nlay, nbnd))
    taussa = np.zeros((ncol, nlay, nbnd))
    taussag = np.zeros((ncol, nlay, nbnd))

    for ibnd in range(nbnd):
        for ilay in range(nlay):
            for icol in range(ncol):
                if mask[icol,ilay]:
                    # Find index into size regime table
                    # This works only if there are precisely three size regimes (four bounds) and it's
                    # previously guaranteed that size_bounds(1) <= size <= size_bounds(4)
                    
                    irad = min(int(np.floor((re[icol,ilay] - re_bounds_ext[1])/re_bounds_ext[2])) + 1, 2)
                    t = lwp[icol,ilay] * \
                        pade_eval(ibnd, nbnd, nsizes, m_ext, n_ext, irad, re[icol,ilay], coeffs_ext)

                    irad = min(int(np.floor((re[icol,ilay] - re_bounds_ssa[1])/re_bounds_ssa[2])) + 1, 2)
                    # Pade approximants for co-albedo can sometimes be negative
                    ts = t * (1.0 - max(0.0,
                        pade_eval(ibnd, nbnd, nsizes, m_ssa, n_ssa, irad, re[icol,ilay], coeffs_ssa)))

                    irad = min(int(np.floor((re[icol,ilay] - re_bounds_asy[1])/re_bounds_asy[2])) + 1, 2)
                    taussag[icol,ilay,ibnd] = \
                        ts * \
                        pade_eval(ibnd, nbnd, nsizes, m_asy, n_asy, irad, re[icol,ilay], coeffs_asy)

                    taussa[icol,ilay,ibnd] = ts
                    tau[icol,ilay,ibnd] = t
                else:
                    tau[icol,ilay,ibnd] = 0.0
                    taussa[icol,ilay,ibnd] = 0.0 
                    taussag[icol,ilay,ibnd] = 0.0

    return tau, taussa, taussag


def pade_eval_nbnd(nbnd, nrads, m, n, irad, re, pade_coeffs):
    """
    Evaluate Padé approximant of order [m/n] for multiple bands.
    
    Args:
        nbnd (int): Number of bands
        nrads (int): Number of radii
        m (int): Order of numerator
        n (int): Order of denominator
        irad (int): Radius index
        re (float): Effective radius
        pade_coeffs (ndarray): Coefficients array with shape (nbnd, nrads, m+n+1)
        
    Returns:
        ndarray: Evaluated Padé approximant for each band
    """
    pade_eval = np.zeros(nbnd)
    
    for iband in range(nbnd):
        # Calculate denominator
        denom = pade_coeffs[iband,irad,n+m]
        for i in range(n+m-1, m, -1):
            denom = pade_coeffs[iband,irad,i] + re*denom
        denom = 1.0 + re*denom
        
        # Calculate numerator
        numer = pade_coeffs[iband,irad,m]
        for i in range(m-1, 0, -1):
            numer = pade_coeffs[iband,irad,i] + re*numer
        numer = pade_coeffs[iband,irad,0] + re*numer
        
        pade_eval[iband] = numer/denom
        
    return pade_eval

def pade_eval_1(iband, nbnd, nrads, m, n, irad, re, pade_coeffs):
    """
    Evaluate Padé approximant of order [m/n] for a single band.
    
    Args:
        iband (int): Band index
        nbnd (int): Number of bands
        nrads (int): Number of radii
        m (int): Order of numerator
        n (int): Order of denominator
        irad (int): Radius index
        re (float): Effective radius
        pade_coeffs (ndarray): Coefficients array with shape (nbnd, nrads, m+n+1)
        
    Returns:
        float: Evaluated Padé approximant for the specified band
    """
    # Calculate denominator
    denom = pade_coeffs[iband,irad,n+m]
    for i in range(n+m-1, m, -1):
        denom = pade_coeffs[iband,irad,i] + re*denom
    denom = 1.0 + re*denom
    
    # Calculate numerator
    numer = pade_coeffs[iband,irad,m]
    for i in range(m-1, 0, -1):
        numer = pade_coeffs[iband,irad,i] + re*numer
    numer = pade_coeffs[iband,irad,0] + re*numer
    
    return numer/denom

# Create a unified interface similar to Fortran's interface
def pade_eval(iband=None, nbnd=None, nrads=None, m=None, n=None, irad=None, re=None, pade_coeffs=None):
    """
    Unified interface for Padé approximant evaluation.
    Calls either pade_eval_nbnd or pade_eval_1 based on whether iband is provided.
    """
    if iband is None:
        return pade_eval_nbnd(nbnd, nrads, m, n, irad, re, pade_coeffs)
    else:
        return pade_eval_1(iband, nbnd, nrads, m, n, irad, re, pade_coeffs)


def compute_cloud_optics(lwp, iwp, rel, rei, cloud_optics):
    """
    Compute cloud optical properties for liquid and ice clouds.
    
    Args:
        lwp (ndarray): Liquid water path (g/m2) with shape (ncol, nlay)
        iwp (ndarray): Ice water path (g/m2) with shape (ncol, nlay)
        rel (ndarray): Liquid effective radius (microns) with shape (ncol, nlay)
        rei (ndarray): Ice effective radius (microns) with shape (ncol, nlay)
        
    Returns:
        tuple: Arrays of optical properties for both liquid and ice phases
    """
    # Get dimensions
    ncol, nlay = lwp.shape
    
    # Create cloud masks
    liq_mask = lwp > 0
    ice_mask = iwp > 0
    
    # Check if cloud optics data is initialized
    if not hasattr(cloud_optics, 'lut_extliq') and not hasattr(cloud_optics, 'pade_extliq'):
        raise ValueError('Cloud optics: no data has been initialized')
        
    # Validate particle sizes are within bounds
    if np.any((rel[liq_mask] < cloud_optics.radliq_lwr.values) | 
              (rel[liq_mask] > cloud_optics.radliq_upr.values)):
        raise ValueError('Cloud optics: liquid effective radius is out of bounds')
        
    if np.any((rei[ice_mask] < cloud_optics.radice_lwr.values) | 
              (rei[ice_mask] > cloud_optics.radice_upr.values)):
        raise ValueError('Cloud optics: ice effective radius is out of bounds')
        
    # Check for negative water paths
    if np.any(lwp[liq_mask] < 0) or np.any(iwp[ice_mask] < 0):
        raise ValueError('Cloud optics: negative lwp or iwp where clouds are supposed to be')

    nbnd = cloud_optics.sizes["nband"]

    # Compute optical properties using lookup tables if available
    if hasattr(cloud_optics, 'lut_extliq'):
        # Liquid phase
        step_size = (cloud_optics.radliq_upr - cloud_optics.radliq_lwr)/(cloud_optics.sizes["nsize_liq"]-1)
        
        ltau, ltaussa, ltaussag = compute_all_from_table(
            ncol, nlay, nbnd, liq_mask, lwp, rel,
            cloud_optics.sizes["nsize_liq"], step_size.values, cloud_optics.radliq_lwr.values,
            cloud_optics.lut_extliq.T, cloud_optics.lut_ssaliq.T, cloud_optics.lut_asyliq.T
        )
        
        # Ice phase
        step_size = (cloud_optics.radice_upr - cloud_optics.radice_lwr)/(cloud_optics.sizes["nsize_ice"]-1)
        ice_roughness = 1
        # [:,:,ice_roughness]
        itau, itaussa, itaussag = compute_all_from_table(
            ncol, nlay, nbnd, ice_mask, iwp, rei,
            cloud_optics.sizes["nsize_ice"], step_size.values, cloud_optics.radice_lwr.values,
            cloud_optics.lut_extice[ice_roughness,:,:].T, cloud_optics.lut_ssaice[ice_roughness,:,:].T, cloud_optics.lut_asyice[ice_roughness,:,:].T
        )
        
    # Otherwise use Pade approximants
    else:
        nsizereg = cloud_optics.pade_extliq.shape[1]
        
        # Liquid phase
        ltau, ltaussa, ltaussag = compute_all_from_pade(
            ncol, nlay, nbnd, nsizereg,
            liq_mask, lwp, rel,
            2, 3, cloud_optics.pade_sizreg_extliq, cloud_optics.pade_extliq,
            2, 2, cloud_optics.pade_sizreg_ssaliq, cloud_optics.pade_ssaliq,
            2, 2, cloud_optics.pade_sizreg_asyliq, cloud_optics.pade_asyliq
        )
        
        # Ice phase  
        itau, itaussa, itaussag = compute_all_from_pade(
            ncol, nlay, nbnd, nsizereg,
            ice_mask, iwp, rei,
            2, 3, cloud_optics.pade_sizreg_extice, cloud_optics.pade_extice,
            2, 2, cloud_optics.pade_sizreg_ssaice, cloud_optics.pade_ssaice,
            2, 2, cloud_optics.pade_sizreg_asyice, cloud_optics.pade_asyice
        )

    # Combine liquid and ice contributions
    tau = ltau + itau
    taussa = ltaussa + itaussa
    taussag = ltaussag + itaussag
    
    # Calculate derived quantities
    ssa = np.divide(taussa, tau, out=np.zeros_like(tau), where=tau > np.finfo(float).eps)
    g = np.divide(taussag, taussa, out=np.zeros_like(tau), where=taussa > np.finfo(float).eps)
    
    return tau, ssa, g