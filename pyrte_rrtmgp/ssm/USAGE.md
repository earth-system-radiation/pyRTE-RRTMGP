# The Simple Spectral Model (SSM)

See [Williams (2026), *Bridging clarity and accuracy: A simple spectral
longwave radiation scheme for idealized climate modeling*](https://doi.org/10.1029/2025MS005405)
and [Czarnecki and Pincus (2026), *How clear-sky spectral overlap shapes
radiation in cloudy atmospheres*](https://journals.ametsoc.org/view/journals/clim/39/14/JCLI-D-25-0589.1.xml)
for the model formulation and parameters, and the Fortran implementation in
[rte-rrtmgp `mo_optics_ssm.F90`](https://github.com/earth-system-radiation/rte-rrtmgp)
for the reference version this Python port reproduces.


## The model in two equations

Each absorbing feature (a *tag*) contributes a mass absorption coefficient
that decays exponentially away from a central wavenumber, so that
log&nbsp;&kappa; is triangle-shaped in wavenumber space:

```
kappa(nu) = kappa0 * exp(-|nu - nu0| / l)
```

The optical depth of each layer is the pressure-scaled sum over tags:

```
tau(nu) = (p_layer / pref) * sum_tags[ layer_mass * kappa(nu) ]
```

Planck sources are `B_nu(T) * dnus` evaluated on the same wavenumber grid, and
broadband fluxes are quadrature sums over that grid.

## Quick start: the default configuration

{class}`~pyrte_rrtmgp.ssm.GasOptics` has no required arguments. Called with
none, it reproduces the longwave defaults of the Fortran RTE-SSM
(`configure_with_defaults` in `mo_optics_ssm.F90`):

```python
from pyrte_rrtmgp.ssm import GasOptics

ssm = GasOptics()
```

| argument | default | value |
|----------|---------|-------|
| `spectral_data` | {data}`~pyrte_rrtmgp.ssm.SSM_W26` | `co2`, `h2o-rot`, `h2o-vr` triangles |
| `nus` | `NUS_LW_DEF` | `linspace(50, 3000, 41)` cm⁻¹ |
| `dnus` | `band_widths(nus)` | edges midway between points, outer edges at 0 and 3500 cm⁻¹ |
| `pref` | `spectral_data.attrs["pref"]` | 50000 Pa |

Arguments override individually. `pref` is read from whichever spectroscopy is
in use, and `dnus` is derived from whichever `nus` is in use, so a custom grid
still gets matching widths:

```python
ssm = GasOptics(spectral_data=SSM_CP26)   # pref = 100022 Pa, taken from the dataset
ssm = GasOptics(nus=my_grid)              # dnus derived to match my_grid
```

This configures the *model*; a call to `compute()` still needs an atmosphere
(see step 4 below). The sections that follow describe each argument in full.

## Workflow

### 1. Configure the spectroscopy (`triangles`)

The spectroscopy is an `xarray.Dataset` holding one variable, `triangles`,
with dimensions `("tags", "params")`. The three params configure each feature:

| param    | meaning                                     | units    | constraint |
|----------|---------------------------------------------|----------|------------|
| `nu0`    | central wavenumber of the feature           | cm⁻¹     | —          |
| `l`      | spectral decay length away from `nu0`       | cm⁻¹     | > 0        |
| `kappa0` | peak mass absorption coefficient at `pref`  | m² kg⁻¹  | ≥ 0        |

Tag names identify the gas: the text before the first hyphen is the species
(`"h2o-rot"` → `h2o`), which must be one of `h2o`, `co2`, or `o3`. A gas may
own several tags (for example separate rotational and vibrational bands).

Two ready-made spectroscopies ship with the module:

* {data}`~pyrte_rrtmgp.ssm.SSM_W26` — Williams (2026); identical to the
  longwave default (`triangle_params_def_lw`) of the Fortran RTE-SSM.
  `pref` attribute: 50000.0 Pa.

  | tag       | nu0 (cm⁻¹) | l (cm⁻¹) | kappa0 (m² kg⁻¹) |
  |-----------|-----------|----------|------------------|
  | `co2`     | 667       | 12       | 110              |
  | `h2o-rot` | 0         | 64       | 282              |
  | `h2o-vr`  | 1600      | 52       | 24               |

* {data}`~pyrte_rrtmgp.ssm.SSM_CP26` — Czarnecki and Pincus (2026), which
  adds an `h2o-cont` continuum tag. `pref` attribute: 100022.0 Pa.

  | tag        | nu0 (cm⁻¹) | l (cm⁻¹) | kappa0 (m² kg⁻¹) |
  |------------|-----------|----------|------------------|
  | `co2`      | 667.5     | 10.2     | 500              |
  | `h2o-rot`  | 150       | 58       | 165              |
  | `h2o-vr`   | 1500      | 60       | 15               |
  | `h2o-cont` | 700       | 275      | 0.1              |

### 2. Choose a spectral grid (`nus`, `dnus`)

`nus` are the wavenumbers (cm⁻¹, strictly increasing) at which absorption and
the Planck function are evaluated; `dnus` is the spectral width (cm⁻¹,
positive) credited to each point. Together they control spectral resolution
and total band coverage. The Fortran RTE-SSM default uses 41 points from 50 to
3000 cm⁻¹, with band edges at the midpoints between points and the outer edges
at 0 and 3500 cm⁻¹:

```python
nus_v = np.linspace(50.0, 3000.0, 41)
mids = 0.5 * (nus_v[:-1] + nus_v[1:])
dnus_v = np.diff(np.concatenate([[0.0], mids, [3500.0]]))

nus = xr.DataArray(nus_v, dims="gpt")
dnus = xr.DataArray(dnus_v, dims="gpt")
```

### 3. Initialize {class}`~pyrte_rrtmgp.ssm.GasOptics`

Spelling out every argument, which is what `GasOptics()` fills in for you:

```python
from pyrte_rrtmgp.ssm import GasOptics, SSM_W26

ssm = GasOptics(
    spectral_data=SSM_W26,
    nus=nus,
    dnus=dnus,
    pref=SSM_W26.attrs["pref"],
)
```

```{note}
All pressures are in **Pa**: the constructor's `pref` argument, the `pref`
attribute on `SSM_W26` (50000.0) and `SSM_CP26` (100022.0), the `P_REF`
default, and the `pres_layer` / `pres_level` inputs. A stored `pref` is passed
straight through with no unit conversion.
```

### 4. Define the atmosphere

The input Dataset uses the same variable names as
{class}`pyrte_rrtmgp.rrtmgp.GasOptics`:

* `pres_layer`, `pres_level` — pressure at layer centers and boundaries (Pa)
* `temp_layer`, `temp_level` — temperature at layer centers and boundaries (K)
* `surface_temperature` — skin temperature (K), no vertical dimension
* one volume mixing ratio variable per species used by the spectroscopy
  (e.g. `h2o`, `co2`), in mol/mol. Well-mixed gases may be a scalar per
  column; they are broadcast across layers.

Levels are layer boundaries, so the `level` dimension is one element longer
than `layer`. Pressure may increase or decrease with index; the orientation is
detected automatically and recorded in the `top_at_1` attribute of the output.

```{warning}
The vertical (`layer`/`level`) axis must be the **last** dimension of every
variable. For multi-column data put the column dimension first, e.g.
`atm = atm.transpose("col", ...)`.
```

### 5. Compute optics with `compute()`

```python
optics = ssm.compute(atm, add_to_input=False)
```

The result contains `tau`, `layer_source`, `level_source`, `surface_source`,
`surface_source_jacobian`, and the spectral grid (`nus`, `dnus`) — everything
the longwave solver needs. With `add_to_input=True` the fields are written
into the input Dataset in place and `None` is returned.

### 6. Solve the radiative transfer equations

Merge in a broadband `surface_emissivity` and call the usual RTE accessor:

```python
problem = xr.merge([optics, atm.surface_emissivity])
fluxes = problem.rte.solve(add_to_input=False)
```

`fluxes` contains `lw_flux_up` and `lw_flux_down` on levels, in W/m².

## Worked example

A complete script solving the radiative-convective equilibrium (RCE) states
end to end. `rce-states.nc` comes from the
[rte-examples](https://github.com/earth-system-radiation/rte-examples)
collection of homogenized atmospheric conditions; it holds 2 variants ×
32 columns × 127 layers, with surface temperatures spanning 273–304 K.

```python
import numpy as np
import xarray as xr

import pyrte_rrtmgp.rte  # noqa: F401  (registers the .rte accessor)
from pyrte_rrtmgp.ssm import SSM_W26, GasOptics

# --- 1. Spectral grid: 41 points from 50 to 3000 cm-1, band edges at the
#        midpoints between points, with the outer edges at 0 and 3500 cm-1.
nus_v = np.linspace(50.0, 3000.0, 41)
mids = 0.5 * (nus_v[:-1] + nus_v[1:])
dnus_v = np.diff(np.concatenate([[0.0], mids, [3500.0]]))

nus = xr.DataArray(nus_v, dims="gpt")
dnus = xr.DataArray(dnus_v, dims="gpt")

# --- 2. Gas optics from the bundled Williams (2026) triangles. Steps 1 and 2
#        spell out the defaults, so a bare GasOptics() is equivalent here.
ssm = GasOptics(
    spectral_data=SSM_W26,
    nus=nus,
    dnus=dnus,
    pref=SSM_W26.attrs["pref"],
)

# --- 3. Radiative-convective equilibrium states: 2 variants x 32 columns
#        x 127 layers. The file stores the vertical axis first, so transpose
#        the column dimension to the front: compute() needs layer/level last.
states = xr.open_dataset("rce-states.nc")
atm = states.isel(variant=0).transpose("col", ...)

print(f"{atm.sizes['col']} columns x {atm.sizes['layer']} layers")
print(
    f"surface temperature: {float(atm.surface_temperature.min()):.1f} - "
    f"{float(atm.surface_temperature.max()):.1f} K"
)

# --- 4. Optical depth + Planck sources, then solve the RTE.
optics = ssm.compute(atm, add_to_input=False)
problem = xr.merge([optics, atm.surface_emissivity])
fluxes = problem.rte.solve(add_to_input=False)

# --- 5. Pressure decreases with index in this file, so level 0 is the
#        surface and the last level is the top of atmosphere.
net = fluxes.lw_flux_up - fluxes.lw_flux_down
net_toa = net.isel(level=-1)
net_sfc = net.isel(level=0)

print(f"net TOA LW: {float(net_toa.min()):6.1f} - {float(net_toa.max()):6.1f} W/m2")
print(f"net SFC LW: {float(net_sfc.min()):6.1f} - {float(net_sfc.max()):6.1f} W/m2")
```

Output:

```
32 columns x 127 layers
surface temperature: 273.0 - 304.0 K
net TOA LW:  241.5 -  335.2 W/m2
net SFC LW:  129.1 -  143.1 W/m2
```

Net outgoing longwave at the top of atmosphere rises steeply with surface
temperature across the RCE ensemble, while the net loss at the surface stays
within a narrow band — the warmer columns are also moister, so the extra
surface emission is largely offset by increased downwelling from the
water-vapor greenhouse.

```{note}
This file stores its dimensions as `(layer, col)`, vertical axis first, so the
`.transpose("col", ...)` in step 3 is required. Without it the kernels read
`col` as the vertical dimension and raise an alignment error.
```

## Validation notes

* This Python implementation reproduces the Fortran RTE-SSM reference fluxes
  to better than 0.04 W/m² (relative differences below 0.02%) across the
  RFMIP, RCE, and CKDMIP example atmospheres.

Results Comparison between Python SSM and Fortran SSM
| case | quantity | N | abs_RMS (W/m2) | abs_max_difference (W/m2) | rel_RMS% | rel_max_difference% |
| --- | --- | --- | --- | --- | --- | --- |
| rfmip | net TOA | 1800 | 0.012591 | 0.018680 | 0.004135 | 0.005682 |
| rfmip | net SFC | 1800 | 0.016802 | 0.025527 | 0.013142 | 0.021726 |
| rce | net TOA | 64 | 0.014082 | 0.018482 | 0.004868 | 0.005514 |
| rce | net SFC | 64 | 0.017764 | 0.019612 | 0.013321 | 0.015209 |
| ckdmip | net TOA | 400 | 0.011238 | 0.025155 | 0.003799 | 0.005841 |
| ckdmip | net SFC | 400 | 0.017002 | 0.031901 | 0.014548 | 0.030962 |
