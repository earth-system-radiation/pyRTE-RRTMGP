#! /usr/env/python
"""Download sample data on which to exercise pyRTE."""
#
#   Data comes from the NASA/GMAO contribution to DYAMOND2 described at
#   https://gmao.gsfc.nasa.gov/global_mesoscale/dyamond_phaseII/data_access/
#   Data is on a cubed sphere at resolution c2880
#   (~3.5 km globally, or about 50 million columns). Files are about
#   6Gb per snapshot (we choose Feb 1 at 9 Z); files for cloud variables are smaller
#
from pathlib import Path

import pooch

compute_gas_optics = False

ymd = "20200201"
ym = ymd[:-2]
sim = "DYAMONDv2_c2880_L181"

urls = [
    f"https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/{sim}/"
    + f"inst_01hr_3d_{v}_Mv/{ym}/{sim}.inst_01hr_3d_{v}_Mv.{ymd}_0900z.nc4"
    for v in ["QL", "QI", "RL", "RI", "DELP", "P", "CO2", "QV", "T"]
]

if compute_gas_optics:
    for v in  ["P", "CO2", "QV", "T"]:
      urls.append(
        f"https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/{sim}/"
        + f"inst_01hr_3d_{v}_Mv/{ym}/{sim}.inst_01hr_3d_{v}_Mv.{ymd}_0900z.nc4")
    #
    # Ozone is 6-hourly
    #
    urls.append(
        f"https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/{sim}/"
        + f"geosgcm_prog/{ym}/{sim}.geosgcm_prog.{ymd}_0600z.nc4"
    )

data_dir = "GEOS-DYAMOND2-data"
with open("registry.txt", "w") as registry:
    for u in urls:
        fname = Path(u).parts[-1]
        # Download each data file to the specified directory
        path = pooch.retrieve(url=u, known_hash=None, fname=fname, path=data_dir)
        # Add the name, hash, and url of the file to the new registry file
        registry.write(f"{fname} {pooch.file_hash(path)} {u}\n")
