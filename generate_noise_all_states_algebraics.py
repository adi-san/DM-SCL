import pandas as pd
import numpy as np
from scipy.optimize import fsolve, least_squares

np.random.seed(0)  # fix the noise so outputs are reproducible across runs

params = {
    'F0': 10.0, 'CA0': 1.0, 'T0': 360.0, 'Fpg': 0.2, 'Fperm': 5.5,
    'rho': 1000.0, 'Cp': 4.184, 'rhoc': 1000.0, 'Cpc': 4.184, 'rhoH': 1000.0, 'CpH': 4.184,
    'dH1': -50000.0, 'dH2': -30000.0, 'A1': 1e6, 'EA1': 40000.0, 'A2': 5e5, 'EA2': 35000.0, 'R': 8.314,
    'dGA': 1000.0, 'dGB': 2000.0, 'dGC': 3000.0,
    'VjR': 2.0, 'FjR': 20.0, 'TCin': 290.0, 'VjS': 1.5, 'FH': 10.0, 'THin': 400.0,
    'UaR': 5000.0, 'UaS': 4000.0, 'kvR': 2.5, 'kvS': 2.0,
}

# read the simulator data from the csv file
data = pd.read_csv('simulator_data_no_noise.csv')
data_copy = data.copy()
data_copy_2=data.copy()
data_copy_3=data.copy()

noise_level = 0.002 # 0.2% noise ensures that temp measurements stay between 2K of actual
# add noise to the data for all columns except F0, FjR, and FH
columns_to_add_noise = [col for col in data.columns if col not in ['F0', 'FjR', 'FH']]
for col in columns_to_add_noise:
    # ensure that noise is generated homoscedastically, here it is heteroscedastic
    min_col = data[col].min()
    max_col = data[col].max()
    ref_magnitude=min_col/2+max_col/2
    data[col] += ref_magnitude * noise_level * np.random.randn(len(data))

# calculate the constitutive terms from the noisy data using skeleton eqns
theta_QR = -params['rhoc']*params['Cpc']*data['FjR']*(params['TCin']-data['TjR'])
theta_QS = params['rhoH']*params['CpH']*data['FH']*(params['THin']-data['TjS'])
r1 = (data['F0']*(params['CA0']-data['CA1']) + data['FR']*(data['CAPerm']-data['CA1']))/data['VR']
r2 = (data['F0']*data['CC1']-data['FR']*(data['CCPerm']-data['CC1']))/data['VR']
K_A = data['CAPerm']/data['CARet']
K_B = data['CBPerm']/data['CBRet']
K_C = data['CCPerm']/data['CCRet']
# create a new dataframe with the noisy states and algebraics and the constitutive terms
data['theta_QR'] = theta_QR
data['theta_QS'] = theta_QS
data['r1'] = r1
data['r2'] = r2
data['K_A'] = K_A
data['K_B'] = K_B
data['K_C'] = K_C
data.to_csv('noisy_states_algebraics_for_all.csv', index=False)

# make parity plots of all the columns with x axes being non-noisy from simulation and y_axes being their corresponding values in data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data_copy is still the clean simulation at this point (noise was added to data in place)
parity_vars_all = list(data.columns)
n_cols_plot = 7
n_rows_plot = int(np.ceil(len(parity_vars_all) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4 * n_cols_plot, 3.6 * n_rows_plot))
for ax, var in zip(axes.ravel(), parity_vars_all):
    x = data_copy[var].to_numpy(dtype=float)
    y = data[var].to_numpy(dtype=float)
    ss_tot = np.sum((x - x.mean()) ** 2)
    r2_val = 1 - np.sum((x - y) ** 2) / ss_tot if ss_tot > 0 else np.nan
    ax.plot(x, y, 'o', ms=3, alpha=0.7)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--')
    ax.set_xlabel(f'simulated {var} (no noise)')
    ax.set_ylabel(f'{var} (noisy)')
    ax.set_title(f'{var}  (R² = {r2_val:.4f})')
for ax in axes.ravel()[len(parity_vars_all):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('noisy_states_algebraics_for_all_parity.png', dpi=120)
plt.close(fig)
print("Saved noisy_states_algebraics_for_all_parity.png")

'''
With DM, we deem that VR, VS, F1, FR, and FRet are well-det and so they are not noisy. The rest of the states and algebraics are noisy.
'''
columns_to_add_noise_DM = [col for col in data_copy.columns if col not in ['F0', 'FjR', 'FH', 'VR', 'VS', 'F1', 'FR', 'FRet']]
for col in columns_to_add_noise_DM:
    min_col = data_copy[col].min()
    max_col = data_copy[col].max()
    ref_magnitude=min_col/2+max_col/2
    data_copy[col] += ref_magnitude * noise_level * np.random.randn(len(data_copy))

data_copy_2=data_copy.copy()
data_copy_3=data_copy.copy()
# data_copy.to_csv('noisy_states_algebraics_DM.csv', index=False)
# in data_copy, delete the columns TjS, CBRet, CCRet, CAPerm, CBPerm, CCPerm
'''
We measure T1, TjR, TS, CC1, CARet, CBRet and CCRet. Using These measurements along with VR, VS, F1, FR, and FRet, 
we can compute the constitutive terms theta_QR, theta_QS, r1, r2, K_A, K_B, and K_C. 
AND the rest of the other states and algebraics.
We need to define a root solver to compute the rest of the states and algebraics from the measured states and algebraics and the constitutive terms.
We will use the skeleton equations to compute the rest of the states and algebraics from the measured states and algebraics and the constitutive terms. The skeleton equations are the same as the ones used in the simulator.
'''
# function used to solve for unknown algebraics, states, and constitutives given DM decomp with measurements
def ss_for_DM(y, p):
    TjS, CA1, CB1, CAPerm, CBPerm, CCPerm, r1, r2, theta_QR, theta_QS, K_A, K_B, K_C = y
    # if something is measured, it is a parameter for a given solve
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    FjR=p['FjR']
    FH=p['FH']


    CC1=p['CC1']
    T1=p['T1']
    TS=p['TS']
    TjR=p['TjR']

    
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']
    

    dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    dT1_dt = ((F0 * (p['T0'] - T1) + FR * (TS - T1)) / VR 
              + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp']) 
              - theta_QR / (p['rho'] * p['Cp'] * VR))
    dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])

    dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    K_A_diff=CAPerm-K_A*CARet
    K_B_diff=CBPerm-K_B*CBRet
    K_C_diff=CCPerm-K_C*CCRet

    output=[dCA1_dt,dCB1_dt,dCC1_dt,dT1_dt,dTjR_dt,dCARet_dt,dCBRet_dt,dCCRet_dt,dTS_dt,dTjS_dt,K_A_diff,K_B_diff,K_C_diff]

    return output
def ss_for_DM_A_b(p):
    # TjS, CA1, CB1, CAPerm, CBPerm, CCPerm, r1, r2, theta_QR, theta_QS, K_A, K_B, K_C = y
    # if something is measured, it is a parameter for a given solve
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    FjR=p['FjR']
    FH=p['FH']
    T0=p['T0']
    VjR=p['VjR']
    TCin=p['TCin']
    Fpg=p['Fpg']
    FH=p['FH']
    VjS=p['VjS']
    THin=p['THin']


    CC1=p['CC1']
    T1=p['T1']
    TS=p['TS']
    TjR=p['TjR']

    
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']

    dH1=p['dH1']
    dH2=p['dH2']
    rho=p['rho']
    Cp=p['Cp']
    rhoc=p['rhoc']
    Cpc=p['Cpc']
    rhoH=p['rhoH']
    CpH=p['CpH']
    b=np.array([F0*CA0/VR,0,(F0+FR)/VR*CC1,(F0*(T0-T1)+FR*(TS-T1))/VR,FjR/VjR*(TCin-TjR),(FR+Fpg-F1)/VS*CARet,
                (FR+Fpg-F1)/VS*CBRet,(FR+Fpg-F1)/VS*CCRet+F1/VS*CC1,F1/VS*(T1-TS),FH/VjS*THin,0,0,0])
    A=np.array([
	[0, (F0+FR)/VR, 0, -FR/VR, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
	[0, 0, -(F0+FR)/VR, 0, FR/VR, 0, 1, -1, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, FR/VR, 0, 1, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, dH1/(rho*Cp), dH2/(rho*Cp), 0, 0, 0, 1/(rho*Cp*VR), 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/(rhoc*Cpc*VjR), 0], 
	[0, -F1/VS, 0, (FR+Fpg)/VS, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, -F1/VS, 0, (FR+Fpg)/VS, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, (FR+Fpg)/VS, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/(rho*Cp*VS)], 
	[FH/VjS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/(rhoH*CpH*VjS)], 
	[0, 0, 0, -1, 0, 0, 0, 0, CARet, 0, 0, 0, 0], 
	[0, 0, 0, 0, -1, 0, 0, 0, 0, CBRet, 0, 0, 0], 
	[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, CCRet, 0, 0]
    ])
    

    # dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    # dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    # dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    # dT1_dt = ((F0 * (p['T0'] - T1) + FR * (TS - T1)) / VR 
    #           + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp']) 
    #           - theta_QR / (p['rho'] * p['Cp'] * VR))
    # dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])

    # dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    # dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    # dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    # dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    # dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    # K_A_diff=CAPerm-K_A*CARet
    # K_B_diff=CBPerm-K_B*CBRet
    # K_C_diff=CCPerm-K_C*CCRet

    # output=[dCA1_dt,dCB1_dt,dCC1_dt,dT1_dt,dTjR_dt,dCARet_dt,dCBRet_dt,dCCRet_dt,dTS_dt,dTjS_dt,K_A_diff,K_B_diff,K_C_diff]

    return A,b



# solve for the unknowns row by row
unknown_names = ['TjS', 'CA1', 'CB1', 'CAPerm', 'CBPerm', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C', 'theta_QR', 'theta_QS']
results = []
for i, row in data_copy.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row values for states, rough magnitudes for constitutive terms
    y0 = [row['TjS'], row['CA1'], row['CB1'], row['CAPerm'], row['CBPerm'], row['CCPerm'],
          0.1, 0.1, 1e6, 1e6, 0.5, 0.5, 0.5]
    sol = fsolve(ss_for_DM, y0, args=(p,),xtol=1e-16)
    A, b = ss_for_DM_A_b(p)
    # print(np.linalg.cond(A))
    sol=np.linalg.solve(A, b)
    res = dict(zip(unknown_names, sol))
    res['condition number']=np.linalg.cond(A)
    res['rank']=np.linalg.matrix_rank(A)
    # res['residual_norm'] = np.linalg.norm(ss_for_DM(sol, p))
    # res['converged'] = res['residual_norm'] < 1e-6
    results.append(res)

results = pd.DataFrame(results, index=data_copy.index)
# measured columns are kept from data_copy, unknown columns are replaced with solved values
data_DM = data_copy.drop(columns=[c for c in unknown_names if c in data_copy.columns]).join(results)
# print(data_DM[unknown_names + ['converged', 'residual_norm']].to_string())
# print(f"Converged: {data_DM['converged'].sum()}/{len(data_DM)}, max residual norm: {data_DM['residual_norm'].max():.3e}")
data_DM.to_csv('noisy_states_algebraics_DM.csv', index=False)

# man this was an utter failure!
"""Suppose I just measure 3 temperatures and thus fully observe the heat transfer terms"""

def ss_DM_just_heat(y, p):
    # if something is measured, it is a parameter for a given solve

    TjS, theta_QR, theta_QS=y

    FjR=p['FjR']
    F1=p['F1']
    FH=p['FH']
    VS=p['VS']
    TjR=p['TjR']
    
    T1=p['T1']
    TS=p['TS']

    dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])
    dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])

    output=[dTjR_dt,dTS_dt,dTjS_dt]
    return output
def ss_DM_just_heat_with_invar(p):
    
    FjR=p['FjR']
    F1=p['F1']
    FH=p['FH']
    VS=p['VS']
    TjR=p['TjR']
        
    T1=p['T1']
    TS=p['TS']
    TjS=p['rho'] * p['Cp'] * F1/(p['rhoH'] * p['CpH']*FH)*(T1-TS)+p['THin']
    theta_QR=-p['rhoc'] * p['Cpc']*FjR*(p['TCin'] - TjR)
    theta_QS=-p['rho'] * p['Cp'] * F1* (T1-TS)
    return [TjS,theta_QR,theta_QS]

# solve for the heat-only unknowns row by row
unknown_names_heat = ['TjS', 'theta_QR', 'theta_QS']
results_heat = []
for i, row in data_copy_2.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row value for TjS, rough magnitudes for heat transfer terms
    y0 = [row['TjS'], 1e6, 1e6]
    sol = fsolve(ss_DM_just_heat, y0, args=(p,))
    # sol=ss_DM_just_heat_with_invar(p)
    # 
    res = dict(zip(unknown_names_heat, sol))
    print(res)
    res['residual_norm'] = np.linalg.norm(ss_DM_just_heat(sol, p))
    res['converged'] = res['residual_norm'] < 1e-6
    results_heat.append(res)

results_heat = pd.DataFrame(results_heat, index=data_copy_2.index)
# measured columns are kept from data_copy_2, unknown columns are replaced with solved values
data_DM_heat = data_copy_2.drop(columns=[c for c in unknown_names_heat if c in data_copy_2.columns]).join(results_heat)
print(data_DM_heat[unknown_names_heat + ['converged', 'residual_norm']].to_string())
print(f"Converged: {data_DM_heat['converged'].sum()}/{len(data_DM_heat)}, max residual norm: {data_DM_heat['residual_norm'].max():.3e}")
data_DM_heat.to_csv('noisy_heat_DM.csv', index=False)

# make parity plots of T1, TjR, TS, TjS, theta_QR, and theta_QS with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_heat
data_clean = pd.read_csv('simulator_data_no_noise.csv')  # 'data' had noise added in place, so reload the clean simulation
parity_vars_heat = ['T1', 'TjR', 'TS', 'TjS', 'theta_QR', 'theta_QS']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, var in zip(axes.ravel(), parity_vars_heat):
    x = data_clean[var].to_numpy(dtype=float)
    y = data_DM_heat[var].to_numpy(dtype=float)
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    print(f"{var}: R^2 = {r2:.4f}, RMSE = {rmse:.4g}")
    ax.plot(x, y, 'o', ms=4, alpha=0.7)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', label='parity')
    ax.set_xlabel(f'simulated {var} (no noise)')
    ax.set_ylabel(f'DM {var} (noisy)')
    ax.set_title(f'{var}  (R² = {r2:.4f})')
    ax.legend()
plt.tight_layout()
plt.savefig('noisy_heat_DM_parity.png', dpi=150)
plt.close(fig)
print("Saved noisy_heat_DM_parity.png")


def ss_DM_rK(y, p):
    # if something is measured, it is a parameter for a given solve
    CA1, CB1, CC1, CCPerm, r1, r2, K_A, K_B, K_C = y
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    # F1=p['F1']
    # FjR=p['FjR']
    # FH=p['FH']

    # CB1=p['CB1']
    # CA1=p['CA1']
    
    # FjR=p['FjR']
    F1=p['F1']
    # FH=p['FH']
    VS=p['VS']
    # TjR=p['TjR']
    
    # T1=p['T1']
    # TS=p['TS']
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']

    CAPerm = p['CAPerm']
    CBPerm = p['CBPerm']

    dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)

    K_A_diff=CAPerm-K_A*CARet
    K_B_diff=CBPerm-K_B*CBRet
    K_C_diff=CCPerm-K_C*CCRet

    output=[dCA1_dt,dCB1_dt,dCC1_dt,dCARet_dt,dCBRet_dt,dCCRet_dt,K_A_diff,K_B_diff,K_C_diff]
    return output

# solve for the reaction/partition unknowns row by row
# unknown_names_rK = ['CC1', 'CAPerm', 'CBPerm', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C']
unknown_names_rK = ['CA1','CB1','CC1', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C']
results_rK = []
for i, row in data_copy_3.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row values for states, rough magnitudes for constitutive terms
    # y0 = [row['CC1'], row['CAPerm'], row['CBPerm'], row['CCPerm'],
    #       0.1, 0.1, 0.5, 0.5, 0.5]
    y0 = [row['CA1'], row['CB1'], row['CC1'], row['CCPerm'],
              0.1, 0.1, 0.5, 0.5, 0.5]
    sol = fsolve(ss_DM_rK, y0, args=(p,))
    res = dict(zip(unknown_names_rK, sol))
    res['residual_norm'] = np.linalg.norm(ss_DM_rK(sol, p))
    res['converged'] = res['residual_norm'] < 1e-6
    results_rK.append(res)

results_rK = pd.DataFrame(results_rK, index=data_copy_3.index)
# measured columns are kept from data_copy_3, unknown columns are replaced with solved values
# actually retrieves r1 and r2 well
data_DM_rK = data_copy_3.drop(columns=[c for c in unknown_names_rK if c in data_copy_3.columns]).join(results_rK)
print(data_DM_rK[unknown_names_rK + ['converged', 'residual_norm']].to_string())
print(f"Converged: {data_DM_rK['converged'].sum()}/{len(data_DM_rK)}, max residual norm: {data_DM_rK['residual_norm'].max():.3e}")
data_DM_rK.to_csv('noisy_r_kappa_DM.csv', index=False)

# the reaction rate terms are nicely measured with CA1, CB1, along with rets meas, the kappa terms need some work! 

# make parity plots of 'CA1','CB1','CC1', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C' with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_rK
parity_vars_rK = ['CA1', 'CB1', 'CC1', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C']
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
for ax, var in zip(axes.ravel(), parity_vars_rK):
    x = data_clean[var].to_numpy(dtype=float)
    y = data_DM_rK[var].to_numpy(dtype=float)
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    print(f"{var}: R^2 = {r2:.4f}, RMSE = {rmse:.4g}")
    ax.plot(x, y, 'o', ms=4, alpha=0.7)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', label='parity')
    ax.set_xlabel(f'simulated {var} (no noise)')
    ax.set_ylabel(f'DM {var} (noisy)')
    ax.set_title(f'{var}  (R² = {r2:.4f})')
    ax.legend()
plt.tight_layout()
plt.savefig('noisy_r_kappa_DM_parity.png', dpi=150)
plt.close(fig)
print("Saved noisy_r_kappa_DM_parity.png")
