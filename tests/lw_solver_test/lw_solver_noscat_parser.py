#!/usr/bin/env python3
import json

# This is the code that was used to save the output of the lw_solver_noscat function, you can use it as reference for other functions
# open(1, file = 'asd/asd/asd/fortran_data.txt', status = 'new') 
      
# ! Write data to the file
# write(1, *) 'ncol: ', ncol
# write(1, *) 'nlay: ', nlay
# write(1, *) 'ngpt: ', ngpt
# write(1, *) 'top_at_1: ', logical(top_at_1, wl)
# write(1, *) 'nmus: ', n_quad_angs
# write(1, *) 'Ds: ', secants
# write(1, *) 'weights: ', gauss_wts(1:n_quad_angs,n_quad_angs)
# write(1, *) 'tau: ', optical_props%tau
# write(1, *) 'lay_source: ', sources%lay_source
# write(1, *) 'lev_source: ', sources%lev_source
# write(1, *) 'sfc_emis: ', sfc_emis_gpt
# write(1, *) 'sfc_src: ', sources%sfc_source
# write(1, *) 'inc_flux: ', inc_flux_diffuse
# write(1, *) 'flux_up: ', gpt_flux_up
# write(1, *) 'flux_dn: ', gpt_flux_dn
# write(1, *) 'do_broadband: ', do_broadband
# write(1, *) 'broadband_up: ', flux_up_loc
# write(1, *) 'broadband_dn: ', flux_dn_loc
# write(1, *) 'do_Jacobians: ', logical(do_Jacobians, wl)
# write(1, *) 'sfc_srcJac: ', sources%sfc_source_Jac
# write(1, *) 'flux_upJac: ', jacobian
# write(1, *) 'do_rescaling: ', logical(.false., wl)
# write(1, *) 'ssa: ', optical_props%tau
# write(1, *) 'g: ', optical_props%tau

# ! Close the file
# close(1)

def parse_value(value):
    # Handle NaN
    if 'NaN' in value:
        res = float('nan')
    # Handle bool
    elif value in ['T', 'F']:
        res = True if value == 'T' else False
    # Handle float
    elif "." in value:
        res = float(value)
    else:
        try:
            # Handle int
            res = int(value)
        except ValueError:
            # Keeping as string if not an int
            res = value
    return res

def load_data_from_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            assert(len(parts) == 2)
            key = parts[0].strip()
            values = parts[1].strip().split()
            if len(values) == 1:
                data[key] = parse_value(values[0])
            else:
                data[key] = [float(val) for val in values]
    return data

file_path = 'fortran_data.txt'
data = load_data_from_file(file_path)

with open('lw_solver_input.json', 'w') as f:
    json.dump(data, f)