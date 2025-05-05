import numpy as np 

def check_config_params(d: dict) -> dict:
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested_dict = check_config_params(value)
        elif isinstance(value, list):
            if np.any(value):
                new_dict[key] = value
        elif value:
            new_dict[key] = value
        else:
            continue

    return new_dict
