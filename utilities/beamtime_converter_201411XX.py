import math
import numpy as np


NAN = np.nan


def dataset_name(name):
    """"""
    def decorator(func):
        def wrapper(theta_position):
            return name, func(theta_position)
        return wrapper
    return decorator

#@dataset_name("pump_probe_delay")


def get_generic(value):
    """"""
    if isinstance(value, int) or isinstance(value, float):
        return value

    if value.find("not-converged") != -1:
        return NAN
    units = ["V", "pulse"]
    for u in units:
        if value.find(u) != -1:
            value = value.replace(u, "")
    return float(value)


def get_delay_from_pulse(pulse):
    """"""
    magic_factor = 6.67 / 1000.
    return float(pulse) * magic_factor


def get_energy_from_theta(theta_position):
    # Todo: Most probably these variables need to be read out from the control system ...
    try:
        theta_position = float(theta_position.replace("pulse", ""))
    except:
        theta_position = float(theta_position)

    theta_coeff = 25000.  # in [pulse/mm]
    lSinbar = 275.0  # in [mm]
    theta_offset = -15.0053431  # in [degree]
    dd = 6.270832

    theta = math.asin((theta_position / theta_coeff) / lSinbar + math.sin(theta_offset * math.pi / 180.0)) * 180.0 / math.pi - theta_offset
    energy = (12.3984 / (dd)) / math.sin(theta * math.pi / 180.0)

    return energy  # , units


def convert(name, array):
    """"""
    convert_dict = {"energy": np.vectorize(get_energy_from_theta, otypes=[np.float]),
                    "delay": np.vectorize(get_delay_from_pulse, otypes=[np.float])}

    if name not in convert_dict:
        convert_dict[name] = np.vectorize(get_generic)
    return convert_dict[name](array)
