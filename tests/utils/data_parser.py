#!/usr/bin/env python3
import json

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

file_path = 'fortran_data_input.txt'
data = load_data_from_file(file_path)

with open('input.json', 'w') as f:
    json.dump(data, f)