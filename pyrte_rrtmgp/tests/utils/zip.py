import json
import os
import zipfile

ZIP = True

p = os.getcwd()
base_name = p + "/tests/rrtmgp_interpolation_test/interpolation"


if ZIP:
    with zipfile.ZipFile(
        base_name + "_test_data.zip", "w", zipfile.ZIP_DEFLATED
    ) as zip_file:
        zip_file.write("input.json")
        zip_file.write("output.json")
else:
    # Unzip data
    input_data = None
    output_data = None
    with zipfile.ZipFile(base_name + "_test_data.zip") as myzip:
        with myzip.open("input.json") as myfile:
            input_data = json.load(myfile)
        with myzip.open("output.json") as myfile:
            output_data = json.load(myfile)

    # Store to Json for verification
    with open(base_name + "_output.json", "w") as f:
        json.dump(output_data, f)

    with open(base_name + "_input.json", "w") as f:
        json.dump(input_data, f)
