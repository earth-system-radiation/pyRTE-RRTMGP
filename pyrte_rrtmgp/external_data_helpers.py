"""Download sample data on which to exercise pyRTE."""

from datetime import datetime
from pathlib import Path

import pooch


def download_dyamond2_data(
    date: datetime,
    compute_gas_optics: bool = False,
    data_dir: str = "GEOS-DYAMOND2-data",
) -> list[str]:
    """
    Download DYAMOND2 data for a specific date and time from NASA/GMAO.

    Data is on a cubed sphere at resolution c2880 (~3.5 km globally, or about
    50 million columns). Files are about 6Gb per snapshot (we choose Feb 1 at 9 Z);
    files for cloud variables are smaller. More information can be found at
    https://gmao.gsfc.nasa.gov/global_mesoscale/dyamond_phaseII/data_access/

    Parameters:
    -----------
    date : datetime.datetime
        Date and time for which to download data.  The ozone file is downloaded
        at the nearest 6-hour mark.
    compute_gas_optics : bool, optional
        Whether to download additional variables needed for gas optics calculations.
        If True, downloads P, CO2, QV, and T variables in addition to the cloud
        variables.
        Defaults to False.
    data_dir : str, optional
        Directory where downloaded files will be stored. Defaults to
        "GEOS-DYAMOND2-data".

    Returns:
    --------
    list
        Paths to downloaded files.
    """
    # Format date strings
    ymd = date.strftime("%Y%m%d")
    ym = date.strftime("%Y%m")
    hour = date.strftime("%H%M")

    sim = "DYAMONDv2_c2880_L181"
    base_url = f"https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/03KM/{sim}/"

    # Define variables to download
    cloud_vars = ["QL", "QI", "RL", "RI", "DELP"]
    gas_vars = ["P", "CO2", "QV", "T"] if compute_gas_optics else []

    # Create URLs for hourly variables
    urls = []
    for v in cloud_vars + gas_vars:
        urls.append(
            f"{base_url}inst_01hr_3d_{v}_Mv/{ym}/{sim}"
            f".inst_01hr_3d_{v}_Mv.{ymd}_{hour}z.nc4"
        )

    # Add ozone (6-hourly) if computing gas optics
    if compute_gas_optics:
        ozone_hour = f"{(date.hour // 6) * 6:02d}00"
        urls.append(
            f"{base_url}geosgcm_prog/{ym}/{sim}.geosgcm_prog.{ymd}_{ozone_hour}z.nc4"
        )

    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True, parents=True)

    downloaded_paths = []
    registry_path = Path(data_dir) / "registry.txt"

    # Create or load registry to avoid re-downloading files
    registry_dict = {}
    if registry_path.exists():
        with open(registry_path, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        registry_dict[parts[0]] = parts[1]

    # Create a single downloader instance with progress bar
    downloader = pooch.HTTPDownloader(progressbar=True)

    # Process all files in a single batch
    with open(registry_path, "a") as registry:
        for u in urls:
            fname = Path(u).parts[-1]

            # If file exists and hash matches, pooch will skip the download
            known_hash = registry_dict.get(fname, None)

            path = pooch.retrieve(
                url=u,
                known_hash=known_hash,
                fname=fname,
                path=data_dir,
                downloader=downloader,
            )
            downloaded_paths.append(path)

            # Only add to registry if it's a new file or hash has changed
            if fname not in registry_dict:
                file_hash = pooch.file_hash(path)
                registry.write(f"{fname} {file_hash} {u}\n")
                registry_dict[fname] = file_hash

    return downloaded_paths
