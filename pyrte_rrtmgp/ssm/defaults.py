# Default constants and triangle parameters from mo_optics_ssm.F90 
# Each triangle is defined by (kappa_0, nu_0, 1)

import numpy as np

TSUN_SSM = 5760.0 #default sun temeprature for SSM
TSI = 1360.0 #default total solar irradiance

#Molecular weights (kg/mol)
MW_H2O = 0.018
MW_CO2 = 0.044
MW_O3 = 0.048
M_DRY = 0.029 #dry air

# default cloud absorption coefficients (m2/kg)
KAPPA_CLD_LW = 50.0
KAPPA_CLD_SW = 0.0001

# default cloud single scattering albedo
SSA_CLD_LW = 0.0
SSA_CLD_SW = 0.9999

# default cloud asymmetry
G_CLD_LW = 0.0
G_CLD_SW = 0.85

# default nnu
NNU_DEF = 41

# wavelength ranges (cm^-1)
NU_MIN_LW_DEF = 0.0
NU_MAX_LW_DEF = 3500.0

NU_MIN_SW_DEF = 0.0
NU_MAX_SW_DEF = 50000.0

# default wavenumber arrays
NUS_LW_DEF = np.linspace(50.0, 3000.0, NNU_DEF)
NUS_SW_DEF = np.linspace(1000.0, 45000.0, NNU_DEF)

# default spectroscopic params
TRIANGLE_PARAMS_DEF_LW = np.array([
  [1., 282., 0., 64.],
  [1., 24., 1600., 52.],
  [2., 110., 667., 12.]
])

GAS_NAMES_DEF_LW = ["h2o", "co2"]

TRIANGLE_PARAMS_DEF_SW = np.array([
  [1., 1., 0., 1200.],
  [2., 0., 0., 1000000.]
]) 

GAS_NAMES_DEF_SW = ["h2o", "o3"]

P_REF = 500.0 * 100.0 # reference pressure hPa -> Pa

MOL_WEIGHTS = {
  "h2o": MW_H2O,
  "co2": MW_CO2,
  "o3": MW_O3,
}
