

import numpy as np


def biomass_regrowth(biogenic_carbon_coeff=100, rotation_period = 18):
    year_vec = np.linspace(0,100,101)

    if rotation_period == 0:
        res_2 = np.zeros(101)

    elif rotation_period == 1:
        res_2 = np.zeros(101)
        res_2[1] = -biogenic_carbon_coeff

    elif rotation_period > 1:
        mu = rotation_period / 2
        sigma = mu / 2
        res_1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((year_vec - mu) / sigma) ** 2) * biogenic_carbon_coeff

        leakage = biogenic_carbon_coeff - res_1.sum()
        perc_contribution = res_1 / biogenic_carbon_coeff
        leakage_to_add = perc_contribution * leakage

        res_2 = -1 * (res_1 + leakage_to_add)
        perc_leakage_2 = 1 - (res_2.sum() / -biogenic_carbon_coeff)

    return res_2
