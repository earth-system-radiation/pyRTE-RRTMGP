
# sigma 

def source(T):
"""Broadband flux from temperature 

    In:
    T [K]:  temperature
        
    Returns:
    source [W/m2-str]: source of broadband intensity (sigma T^4/pi)
    """ 

def set_up_rad_equil(Tsurf, tau_array, D = 1./0.6096748751): 
"""Sets up a temperature profile consistent with grey radiative equillibirum given 
    a surface temperature and an array of optical depth increments.
    Assumes the top-of-atmosphere is at index 0

    In:
    Tsurf [K]: surface temperature
    tau_array []; optical depth increwments 
        
    Returns:
    T [k]: profile of temperature consistent with  grey radiative equillibirum
    """ 

def olr_in_re(Tsurf, total_tau, OLR, D = 1./0.6096748751):
"""Checks that the top-of-atmosphere outgoing longwave radiation 
   is consistent with the optcal depth and surface temperature

    In:
    Tsurf [K]: surface temperature
    total_tau []; total_optical_depth 
    OLR [W/m2]: 
        
    Returns: True/False 
    """ 
    (2._wp * sigma * T**4)/(2 + D total_tau) 

   #
   # Each check requires surface T and an optical depth distribution 
   #
   # Compute fluxes; check OLR and constant net flux 
   #