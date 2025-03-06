#! /usr/env/python 

#
# Download sample data on which to exercise pyRTE. 
#   Data comes from the NASA/GMAO contribution to DYAMOND2 described at 
#   https://gmao.gsfc.nasa.gov/global_mesoscale/dyamond_phaseII/data_access/
#   Data is on a cubed sphere at resolution c2880 (~3.5 km globally, or about 50 million columns)
#   Files are about 6Gb per snapshot (we choose Feb 1 at 9 Z); files for cloud variables are smaller
#

urls = [
  f"https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/DYAMONDv2_c2880_L181/inst_01hr_3d_{v}_Mv/202002/DYAMONDv2_c2880_L181.inst_01hr_3d_{v}_Mv.20200201_0900z.nc4"
  for v in ["QL", "QI", "RL", "RI", "DELP", "CO2", "QV", "T", "P"]
]
#
# Ozone is 6-hourly
#
urls.append(["https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/DYAMONDv2_c2880_L181/geosgcm_prog/202002/DYAMONDv2_c2880_L181.geosgcm_prog.20200201_0600z.nc4"])


import pooch 
from pathlib import Path 

data_dir = "GEOS-DYAMOND2-data"
with open("registry.txt", "w") as registry:
  for u in urls: 
    fname=Path(u).parts[-1]
    # Download each data file to the specified directory
    path = pooch.retrieve(
        url=u, known_hash=None, fname=fname, path=data_dir
    )
    # Add the name, hash, and url of the file to the new registry file
    registry.write(
        f"{fname} {pooch.file_hash(path)} {u}\n"
    )