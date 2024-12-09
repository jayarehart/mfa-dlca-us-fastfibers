# File to create the emisisons inventory from the material flow analysis.
# Note that additional scenarios are considered in this repository that are not included in the manuscript.
# Within the scripts here, the following matching is used to connect scenarios names:
#
#   S0 = S1 BAU
#   S2 = S2 Early-Slow
#   S7 = S3 Delayed-Fast
#   S6 = S4 Optimistic
#
#   On the left is the nomenclature used in this repository. On the right is name used in the manuscript.


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.helpers import biomass_regrowth

# Import excel file for inflow scenarios
file_name = 'excel_analysis/inflow_scenarios.xlsx'
inflow = pd.read_excel(io=file_name,
                       sheet_name='Inflow',
                       index_col=0
                       )
# Ratio of building stock by system type for each scenario (analysis from work done previously in excel)
S0_ratios = pd.read_excel(io=file_name, sheet_name='S0', index_col=0)
# S1_ratios = pd.read_excel(io=file_name, sheet_name='S1', index_col=0)
S2_ratios = pd.read_excel(io=file_name, sheet_name='S2', index_col=0)
# S3_ratios = pd.read_excel(io=file_name, sheet_name='S3', index_col=0)
# S4_ratios = pd.read_excel(io=file_name, sheet_name='S4', index_col=0)
# S5_ratios = pd.read_excel(io=file_name, sheet_name='S5', index_col=0)
S6_ratios = pd.read_excel(io=file_name, sheet_name='S6', index_col=0)
# S4a_ratios = pd.read_excel(io=file_name, sheet_name='S4a', index_col=0)
S7_ratios = pd.read_excel(io=file_name, sheet_name='S7', index_col=0)

# Scale the building stock inflow of floor space by the market share of each system type. Convert units so that final is in Mt
S0_inflow = S0_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
# S1_inflow = S1_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
S2_inflow = S2_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
# S3_inflow = S3_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
# S4_inflow = S4_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
# S5_inflow = S5_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
S6_inflow = S6_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
# S4a_inflow = S4a_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9
S7_inflow = S7_ratios.multiply(inflow["Inflow"], axis=0) * 1e6 / 1e9



def create_emissions_inv(A1_A3, Bio_1, Rot_1, Bio_2=0, Rot_2=0, Bio_3=0, Rot_3=0):
    '''
    Create the emisisons inventory for a system that has three rotation periods
    :param A1_A3: A1-A3 emissions of the system
    :param Bio_1: Biogenic carbon uptake (positive number) for first rotation period
    :param Rot_1: First rotation period
    :param Bio_2: Biogenic carbon uptake (positive number) for second rotation period
    :param Rot_2: Second rotation period
    :param Bio_3: Biogenic carbon uptake (positive number) for third rotation period
    :param Rot_3: Third rotation period
    :return: The emissions flux for the system
    '''

    year_vec = np.linspace(0,100,101)
    res = pd.DataFrame({
        'year': year_vec,
    })
    res['A1_A3'] = 0
    res.loc[0, 'A1_A3'] = A1_A3

    uptake_1 = biomass_regrowth(Bio_1, Rot_1)
    res['Uptake_1'] = uptake_1

    uptake_2 = biomass_regrowth(Bio_2, Rot_2)
    res['Uptake_2'] = uptake_2

    uptake_3 = biomass_regrowth(Bio_3, Rot_3)
    res['Uptake_3'] = uptake_2

    res['Total'] = res['A1_A3'] + res['Uptake_1'] + res['Uptake_2'] + res['Uptake_3']

    df_zero = pd.DataFrame(0, index=np.arange(400), columns=res.columns)
    res = pd.concat([res, df_zero]).reset_index()

    # res_2 = EmissionsFluxTotal(time=[], CO2_flux=[], CH4_flux=[], N2O_flux=[])
    # res_2.time = year_vec
    # res_2.CO2_flux = res['Total']
    # res_2.CH4_flux = [np.zeros(501, dtype=np.int8)]
    # res_2.N2O_flux = [np.zeros(501)]

    return res


def stock_cohort_create_inv(inflow: S0_inflow):
    # Function to create the emissions inventory for the full building stock (including stock/cohort analysis)

    inv_CMU = create_emissions_inv(69.79, 4.331, 65)
    inv_2x6 = create_emissions_inv(16.22, 14.38, 65)
    inv_hybrid = create_emissions_inv(50.42, 6.66, 65, 20.63, 1, 28.13, 19)
    inv_esc = create_emissions_inv(27.37, 6.66, 65, 17.9, 11, 31.84, 19)

    counter = 0
    inv_2x6_tot = pd.DataFrame()
    inv_CMU_tot = pd.DataFrame()
    inv_hybrid_tot = pd.DataFrame()
    inv_esc_tot = pd.DataFrame()

    for index, row in inflow.iterrows():
        twox6_i = row['2x6'] * inv_2x6['Total']
        CMU_i = row['CMU'] * inv_CMU['Total']
        hybrid_i = row['Hybrid'] * inv_hybrid['Total']
        esc_i = row['ESC'] * inv_esc['Total']
        # shift down one year
        twox6_shift = pd.DataFrame({index: twox6_i}).shift(counter)
        CMU_shift = pd.DataFrame({index: CMU_i}).shift(counter)
        hybrid_shift = pd.DataFrame({index: hybrid_i}).shift(counter)
        esc_shift = pd.DataFrame({index: esc_i}).shift(counter)

        inv_2x6_tot[index] = twox6_shift
        inv_CMU_tot[index] = CMU_shift
        inv_hybrid_tot[index] = hybrid_shift
        inv_esc_tot[index] = esc_shift
        counter = counter + 1

    res = pd.DataFrame({
        'inv_2x6': inv_2x6_tot.sum(axis=1),
        'inv_CMU': inv_CMU_tot.sum(axis=1),
        'inv_hybrid': inv_hybrid_tot.sum(axis=1),
        'inv_esc': inv_esc_tot.sum(axis=1),
    })[0:76]       #[0:76]

    res['total'] = res['inv_2x6'] + res['inv_CMU'] + res['inv_hybrid'] + res['inv_esc']
    res.index +=2025


    return inv_2x6_tot, inv_CMU_tot, inv_hybrid_tot, inv_esc_tot, res


S0_df_2x6, S0_df_CMU, S0_df_hybrid, S0_df_esc, S0_emission_df = stock_cohort_create_inv(S0_inflow)
# S1_df_2x6, S1_df_CMU, S1_df_hybrid, S1_df_esc, S1_emission_df = stock_cohort_create_inv(S1_inflow)
S2_df_2x6, S2_df_CMU, S2_df_hybrid, S2_df_esc, S2_emission_df = stock_cohort_create_inv(S2_inflow)
# S3_df_2x6, S3_df_CMU, S3_df_hybrid, S3_df_esc, S3_emission_df = stock_cohort_create_inv(S3_inflow)
# S4_df_2x6, S4_df_CMU, S4_df_hybrid, S4_df_esc, S4_emission_df = stock_cohort_create_inv(S4_inflow)
# S5_df_2x6, S5_df_CMU, S5_df_hybrid, S5_df_esc, S5_emission_df = stock_cohort_create_inv(S5_inflow)
S6_df_2x6, S6_df_CMU, S6_df_hybrid, S6_df_esc, S6_emission_df = stock_cohort_create_inv(S6_inflow)
# S4a_df_2x6, S4a_df_CMU, S4a_df_hybrid, S4a_df_esc, S4a_emission_df = stock_cohort_create_inv(S4a_inflow)
S7_df_2x6, S7_df_CMU, S7_df_hybrid, S7_df_esc, S7_emission_df = stock_cohort_create_inv(S7_inflow)


S0_emission_df.plot(title='Scenario 0: Annual Emissions [Mt CO2e/year]')
plt.show()
# S1_emission_df.plot(title='Scenario 1: Annual Emissions [Mt CO2e/year]')
# plt.show()
S2_emission_df.plot(title='Scenario 2: Annual Emissions [Mt CO2e/year]')
plt.show()
# S3_emission_df.plot(title='Scenario 3: Annual Emissions [Mt CO2e/year]')
# plt.show()
# S4_emission_df.plot(title='Scenario 4: Annual Emissions [Mt CO2e/year]')
# plt.show()
# S5_emission_df.plot(title='Scenario 5: Annual Emissions [Mt CO2e/year]')
# plt.show()
S6_emission_df.plot(title='Scenario 6: Annual Emissions [Mt CO2e/year]')
plt.show()
# S4a_emission_df.plot(title='Scenario 4a: Annual Emissions [Mt CO2e/year]')
# plt.show()
S7_emission_df.plot(title='Scenario 7: Annual Emissions [Mt CO2e/year]')
plt.show()



with pd.ExcelWriter('results/MFA_Emissions_Output.xlsx') as writer:
    S0_emission_df.to_excel(writer, sheet_name='S0')
    # S1_emission_df.to_excel(writer, sheet_name='S1')
    S2_emission_df.to_excel(writer, sheet_name='S2')
    # S3_emission_df.to_excel(writer, sheet_name='S3')
    # S4_emission_df.to_excel(writer, sheet_name='S4')
    # S5_emission_df.to_excel(writer, sheet_name='S5')
    S6_emission_df.to_excel(writer, sheet_name='S6')
    # S4a_emission_df.to_excel(writer, sheet_name='S4a')
    S6_emission_df.to_excel(writer, sheet_name='S7')

    
    S0_df_2x6.to_excel(writer, sheet_name='S0_2x6_SC')
    S0_df_CMU.to_excel(writer, sheet_name='S0_CMU_SC')
    S0_df_hybrid.to_excel(writer, sheet_name='S0_Hybrid_SC')
    S0_df_esc.to_excel(writer, sheet_name='S0_ESC_SC')

    # S1_df_2x6.to_excel(writer, sheet_name='S1_2x6_SC')
    # S1_df_CMU.to_excel(writer, sheet_name='S1_CMU_SC')
    # S1_df_hybrid.to_excel(writer, sheet_name='S1_Hybrid_SC')
    # S1_df_esc.to_excel(writer, sheet_name='S1_ESC_SC')
    
    S2_df_2x6.to_excel(writer, sheet_name='S2_2x6_SC')
    S2_df_CMU.to_excel(writer, sheet_name='S2_CMU_SC')
    S2_df_hybrid.to_excel(writer, sheet_name='S2_Hybrid_SC')
    S2_df_esc.to_excel(writer, sheet_name='S2_ESC_SC')
    
    # S3_df_2x6.to_excel(writer, sheet_name='S3_2x6_SC')
    # S3_df_CMU.to_excel(writer, sheet_name='S3_CMU_SC')
    # S3_df_hybrid.to_excel(writer, sheet_name='S3_Hybrid_SC')
    # S3_df_esc.to_excel(writer, sheet_name='S3_ESC_SC')
    
    # S4_df_2x6.to_excel(writer, sheet_name='S4_2x6_SC')
    # S4_df_CMU.to_excel(writer, sheet_name='S4_CMU_SC')
    # S4_df_hybrid.to_excel(writer, sheet_name='S4_Hybrid_SC')
    # S4_df_esc.to_excel(writer, sheet_name='S4_ESC_SC')
    
    # S5_df_2x6.to_excel(writer, sheet_name='S5_2x6_SC')
    # S5_df_CMU.to_excel(writer, sheet_name='S5_CMU_SC')
    # S5_df_hybrid.to_excel(writer, sheet_name='S5_Hybrid_SC')
    # S5_df_esc.to_excel(writer, sheet_name='S5_ESC_SC')
    
    S6_df_2x6.to_excel(writer, sheet_name='S6_2x6_SC')
    S6_df_CMU.to_excel(writer, sheet_name='S6_CMU_SC')
    S6_df_hybrid.to_excel(writer, sheet_name='S6_Hybrid_SC')
    S6_df_esc.to_excel(writer, sheet_name='S6_ESC_SC')
    
    # S4a_df_2x6.to_excel(writer, sheet_name='S4a_2x6_SC')
    # S4a_df_CMU.to_excel(writer, sheet_name='S4a_CMU_SC')
    # S4a_df_hybrid.to_excel(writer, sheet_name='S4a_Hybrid_SC')
    # S4a_df_esc.to_excel(writer, sheet_name='S4a_ESC_SC')
    
    S7_df_2x6.to_excel(writer, sheet_name='S7_2x6_SC')
    S7_df_CMU.to_excel(writer, sheet_name='S7_CMU_SC')
    S7_df_hybrid.to_excel(writer, sheet_name='S7_Hybrid_SC')
    S7_df_esc.to_excel(writer, sheet_name='S7_ESC_SC')


# Check the mass balance
inv_CMU = create_emissions_inv(69.79, 4.331, 65)['Total'].sum()
inv_2x6 = create_emissions_inv(16.22, 14.38, 65)['Total'].sum()
inv_hybrid = create_emissions_inv(50.42, 6.66, 65, 20.63, 1, 28.13, 19)['Total'].sum()
inv_esc = create_emissions_inv(27.37, 6.66, 65, 17.9, 11, 31.84, 19)['Total'].sum()


S0_inflow['2x6'].sum() * inv_2x6 - S0_emission_df['inv_2x6'].sum()
S0_inflow['CMU'].sum() * inv_CMU - S0_emission_df['inv_CMU'].sum()
S0_inflow['Hybrid'].sum() * inv_hybrid - S0_emission_df['inv_hybrid'].sum()
S0_inflow['ESC'].sum() * inv_esc - S0_emission_df['inv_esc'].sum()


# Write flux file for DLCA input
S0_flux = pd.DataFrame({
    'year': np.linspace(start=0, stop=75, num=76),
    'CO2_flux': S0_emission_df['total'] * 1e9,
    'CH4_flux' : np.linspace(start=0, stop=0, num=76),
    'gas1_flux' : np.linspace(start=0, stop=0, num=76),
    'gas2_flux' : np.linspace(start=0, stop=0, num=76),
    'gas3_flux' : np.linspace(start=0, stop=0, num=76),
    'other_forcing' : np.linspace(start=0, stop=0, num=76)
})

# S1_flux = pd.DataFrame({
#     'year': np.linspace(start=0, stop=75, num=76),
#     'CO2_flux': S1_emission_df['total']* 1e9,
#     'CH4_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas1_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas2_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas3_flux' : np.linspace(start=0, stop=0, num=76),
#     'other_forcing' : np.linspace(start=0, stop=0, num=76)
# })

S2_flux = pd.DataFrame({
    'year': np.linspace(start=0, stop=75, num=76),
    'CO2_flux': S2_emission_df['total']* 1e9,
    'CH4_flux' : np.linspace(start=0, stop=0, num=76),
    'gas1_flux' : np.linspace(start=0, stop=0, num=76),
    'gas2_flux' : np.linspace(start=0, stop=0, num=76),
    'gas3_flux' : np.linspace(start=0, stop=0, num=76),
    'other_forcing' : np.linspace(start=0, stop=0, num=76)
})

# S3_flux = pd.DataFrame({
#     'year': np.linspace(start=0, stop=75, num=76),
#     'CO2_flux': S3_emission_df['total']* 1e9,
#     'CH4_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas1_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas2_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas3_flux' : np.linspace(start=0, stop=0, num=76),
#     'other_forcing' : np.linspace(start=0, stop=0, num=76)
# })

# S4_flux = pd.DataFrame({
#     'year': np.linspace(start=0, stop=75, num=76),
#     'CO2_flux': S4_emission_df['total']* 1e9,
#     'CH4_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas1_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas2_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas3_flux' : np.linspace(start=0, stop=0, num=76),
#     'other_forcing' : np.linspace(start=0, stop=0, num=76)
# })

# S5_flux = pd.DataFrame({
#     'year': np.linspace(start=0, stop=75, num=76),
#     'CO2_flux': S5_emission_df['total']* 1e9,
#     'CH4_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas1_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas2_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas3_flux' : np.linspace(start=0, stop=0, num=76),
#     'other_forcing' : np.linspace(start=0, stop=0, num=76)
# })

S6_flux = pd.DataFrame({
    'year': np.linspace(start=0, stop=75, num=76),
    'CO2_flux': S6_emission_df['total']* 1e9,
    'CH4_flux' : np.linspace(start=0, stop=0, num=76),
    'gas1_flux' : np.linspace(start=0, stop=0, num=76),
    'gas2_flux' : np.linspace(start=0, stop=0, num=76),
    'gas3_flux' : np.linspace(start=0, stop=0, num=76),
    'other_forcing' : np.linspace(start=0, stop=0, num=76)
})

# S4a_flux = pd.DataFrame({
#     'year': np.linspace(start=0, stop=75, num=76),
#     'CO2_flux': S4a_emission_df['total']* 1e9,
#     'CH4_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas1_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas2_flux' : np.linspace(start=0, stop=0, num=76),
#     'gas3_flux' : np.linspace(start=0, stop=0, num=76),
#     'other_forcing' : np.linspace(start=0, stop=0, num=76)
# })

S7_flux = pd.DataFrame({
    'year': np.linspace(start=0, stop=75, num=76),
    'CO2_flux': S7_emission_df['total']* 1e9,
    'CH4_flux' : np.linspace(start=0, stop=0, num=76),
    'gas1_flux' : np.linspace(start=0, stop=0, num=76),
    'gas2_flux' : np.linspace(start=0, stop=0, num=76),
    'gas3_flux' : np.linspace(start=0, stop=0, num=76),
    'other_forcing' : np.linspace(start=0, stop=0, num=76)
})

with pd.ExcelWriter('results/flux_S0.xlsx') as writer:
    S0_flux.to_excel(writer, sheet_name='flux'),
# with pd.ExcelWriter('flux_S1.xlsx') as writer:
#     S1_flux.to_excel(writer, sheet_name='flux'),
with pd.ExcelWriter('results/flux_S2.xlsx') as writer:
    S2_flux.to_excel(writer, sheet_name='flux'),
# with pd.ExcelWriter('flux_S3.xlsx') as writer:
#     S3_flux.to_excel(writer, sheet_name='flux'),
# with pd.ExcelWriter('flux_S4.xlsx') as writer:
#     S4_flux.to_excel(writer, sheet_name='flux'),
# with pd.ExcelWriter('flux_S5.xlsx') as writer:
#     S5_flux.to_excel(writer, sheet_name='flux'),
with pd.ExcelWriter('results/flux_S6.xlsx') as writer:
    S6_flux.to_excel(writer, sheet_name='flux'),
# with pd.ExcelWriter('flux_S4a.xlsx') as writer:
#     S4a_flux.to_excel(writer, sheet_name='flux'),
with pd.ExcelWriter('results/flux_S7.xlsx') as writer:
    S7_flux.to_excel(writer, sheet_name='flux'),
