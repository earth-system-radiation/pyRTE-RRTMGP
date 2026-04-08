# Default constants and triangle parameters from mo_optics_ssm.F90 
# Each triangle is defined by (kappa_0, nu_0, 1)

import numpy as np

Tsun_ssm = 5760.0 #default sun temeprature for SSM
tsi = 1360.0 #default total solar irradiance

#Molecular weights (kg/mol)
mw_h2o = 0.018
mw_co2 = 0.044
mw_03 = 0.048
m_dry = 0.029 #dry air

# default cloud absorption coefficients (m2/kg)
kappa_cld_lw = 50.0
kappa_cld_sw == 0.0001

# default cloud single scattering albedo
ssa_cld_lw = 0.0
ssa_cld_sw = 0.9999

# default cloud asymmetry
g_cld_lw = 0.0
g_cld_sw = 0.85

# default nnu
nnu_def = 41

# wavelength ranges (cm^-1)
nu_min_lw_def = 0.0
nu_max_lw_def = 3500.0

nu_min_sw_def = 0.0
nu_max_sw_def = 50000.0

# default wavenumber arrays
nus_lw_def = np.linspace(50.0, 3000.0, nnu_def)
nus_sw_def = np.linspace(1000.0, 45000.0, nnu_def)



