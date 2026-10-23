import pandas as pd
import numpy as np
from scipy.optimize import fsolve, least_squares
from sklearn.linear_model import Ridge
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

noise_level = 0.002 # 0.2% noise ensures that temp measurements stay between 2K of actual

'''
With DM, we deem that VR, VS, F1, FR, and FRet are well-det and so they are not noisy. The rest of the states and algebraics are noisy.
'''
columns_to_add_noise_DM = [col for col in data_copy.columns if col not in ['F0', 'FjR', 'FH', 'VR', 'VS', 'F1', 'FR', 'FRet']]
noise_sigma_DM = {}  # std dev of the noise added to each column, used to weight least squares fits
for col in columns_to_add_noise_DM:
    min_col = data_copy[col].min()
    max_col = data_copy[col].max()
    ref_magnitude=min_col/2+max_col/2
    noise_sigma_DM[col] = abs(ref_magnitude) * noise_level
    data_copy[col] += ref_magnitude * noise_level * np.random.randn(len(data_copy))

# regenerate noisy_states_algebraics_for_all.csv / _parity.png from the same noisy data used for the DM results
# (data_copy), so noise is not added twice. Only the well-det volumes and flows (noise-free in DM) get noise here;
# F0, FjR, and FH stay noise-free
data_all = data_copy.copy()
columns_to_add_noise_all = ['VR', 'VS', 'F1', 'FR', 'FRet']
for col in columns_to_add_noise_all:
    min_col = data[col].min()
    max_col = data[col].max()
    ref_magnitude=min_col/2+max_col/2
    data_all[col] += ref_magnitude * noise_level * np.random.randn(len(data_all))

# calculate the constitutive terms from the noisy data using skeleton eqns
theta_QR = -params['rhoc']*params['Cpc']*data_all['FjR']*(params['TCin']-data_all['TjR'])
theta_QS = params['rhoH']*params['CpH']*data_all['FH']*(params['THin']-data_all['TjS'])
r1 = (data_all['F0']*(params['CA0']-data_all['CA1']) + data_all['FR']*(data_all['CAPerm']-data_all['CA1']))/data_all['VR']
r2 = (data_all['F0']*data_all['CC1']-data_all['FR']*(data_all['CCPerm']-data_all['CC1']))/data_all['VR']
K_A = data_all['CAPerm']/data_all['CARet']
K_B = data_all['CBPerm']/data_all['CBRet']
K_C = data_all['CCPerm']/data_all['CCRet']
# overwrite the constitutive columns with the values computed from the noisy states and algebraics
data_all['theta_QR'] = theta_QR
data_all['theta_QS'] = theta_QS
data_all['r1'] = r1
data_all['r2'] = r2
data_all['K_A'] = K_A
data_all['K_B'] = K_B
data_all['K_C'] = K_C
data_all.to_csv('noisy_states_algebraics_for_all.csv', index=False)

# make parity plots of all the columns with x axes being non-noisy from simulation and y_axes being their corresponding values in data_all
parity_vars_all = list(data_all.columns)
n_cols_plot = 7
n_rows_plot = int(np.ceil(len(parity_vars_all) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4 * n_cols_plot, 3.6 * n_rows_plot))
for ax, var in zip(axes.ravel(), parity_vars_all):
    x = data[var].to_numpy(dtype=float)
    y = data_all[var].to_numpy(dtype=float)
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

    return A,b

def ruiz_scaling(A, iters=20):
    # iteratively scale rows and columns so the largest |entry| in each is ~1
    Dr = np.ones(A.shape[0])
    Dc = np.ones(A.shape[1])
    M = A.copy()
    for _ in range(iters):
        dr = 1/np.sqrt(np.abs(M).max(axis=1))
        dc = 1/np.sqrt(np.abs(M).max(axis=0))
        M = dr[:, None] * M * dc[None, :]
        Dr *= dr
        Dc *= dc
    return Dr, Dc

# solve for the unknowns row by row
unknown_names = ['TjS', 'CA1', 'CB1', 'CAPerm', 'CBPerm', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C', 'theta_QR', 'theta_QS']
results = []
for i, row in data_copy.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row values for states, rough magnitudes for constitutive terms
    # y0 = [row['TjS'], row['CA1'], row['CB1'], row['CAPerm'], row['CBPerm'], row['CCPerm'],
    #       0.1, 0.1, 1e6, 1e6, 0.5, 0.5, 0.5]
    # sol = fsolve(ss_for_DM, y0, args=(p,),xtol=1e-16)
    A, b = ss_for_DM_A_b(p)
    # print(np.linalg.cond(A))
    # my system is ill-conditioned, I want to see if scaling the rows will help
    # Ruiz equilibration: solve (Dr A Dc) z = Dr b, then x = Dc z
    Dr, Dc = ruiz_scaling(A)
    A_scaled = Dr[:, None] * A * Dc[None, :]
    sol = Dc * np.linalg.solve(A_scaled, Dr * b)
    sol_unscaled = np.linalg.solve(A, b)
    res = dict(zip(unknown_names, sol))
    res['condition number']=np.linalg.cond(A)
    res['condition number scaled']=np.linalg.cond(A_scaled)
    res['max rel diff scaled vs unscaled']=np.max(np.abs(sol-sol_unscaled)/np.abs(sol_unscaled))
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
def ss_for_DM_just_heat_A_b(p):
    FjR=p['FjR']
    F1=p['F1']
    FH=p['FH']
    VS=p['VS']
    TjR=p['TjR']

    T1=p['T1']
    TS=p['TS']
    VjR=p['VjR']
    TCin=p['TCin']
    THin=p['THin']
    rhoc=p['rhoc']
    Cpc=p['Cpc']
    VjS=p['VjS']
    rhoH=p['rhoH']
    CpH=p['CpH']
    rho=p['rho']
    Cp=p['Cp']

    b=np.array([FjR/VjR*(TCin-TjR),F1/VS*(T1-TS),FH/VjS*THin])
    A=np.array([[0,-1/(rhoc*Cpc*VjR),0],
                [0,0,-1/(rho*Cp*VS)],
                [FH/VjS,0,1/(rhoH*CpH*VjS)]
                ])
    return A, b

# solve for the heat-only unknowns row by row
unknown_names_heat = ['TjS', 'theta_QR', 'theta_QS']
results_heat = []
for i, row in data_copy_2.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row value for TjS, rough magnitudes for heat transfer terms
    # y0 = [row['TjS'], 1e6, 1e6]
    # sol = fsolve(ss_DM_just_heat, y0, args=(p,))
    # sol=ss_DM_just_heat_with_invar(p)
    A, b = ss_for_DM_just_heat_A_b(p)
    # 
    Dr, Dc = ruiz_scaling(A)
    A_scaled = Dr[:, None] * A * Dc[None, :]
    sol = Dc * np.linalg.solve(A_scaled, Dr * b)
    sol_unscaled = np.linalg.solve(A, b)
    res = dict(zip(unknown_names_heat, sol))
    res['condition number']=np.linalg.cond(A)
    res['condition number scaled']=np.linalg.cond(A_scaled)
    res['max rel diff scaled vs unscaled']=np.max(np.abs(sol-sol_unscaled)/np.abs(sol_unscaled))
    res['rank']=np.linalg.matrix_rank(A)
        # res['residual_norm'] = np.linalg.norm(ss_for_DM(sol, p))
        # res['converged'] = res['residual_norm'] < 1e-6
    # unknown_names_heat.append(res)
    
    # res = dict(zip(unknown_names_heat, sol))
    # print(res)
    # res['residual_norm'] = np.linalg.norm(ss_DM_just_heat(sol, p))
    # res['converged'] = res['residual_norm'] < 1e-6
    results_heat.append(res)

results_heat = pd.DataFrame(results_heat, index=data_copy_2.index)
# measured columns are kept from data_copy_2, unknown columns are replaced with solved values
data_DM_heat = data_copy_2.drop(columns=[c for c in unknown_names_heat if c in data_copy_2.columns]).join(results_heat)
# print(data_DM_heat[unknown_names_heat + ['converged', 'residual_norm']].to_string())
# print(f"Converged: {data_DM_heat['converged'].sum()}/{len(data_DM_heat)}, max residual norm: {data_DM_heat['residual_norm'].max():.3e}")
data_DM_heat.to_csv('noisy_heat_DM.csv', index=False)

# make parity plots of T1, TjR, TS, TjS, theta_QR, and theta_QS with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_heat
data_clean = data  # 'data' is never modified, so it is the clean simulation
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
    Fpg=p['Fpg']

    dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + Fpg) / VS) * (CARet - CAPerm)
    dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + Fpg) / VS) * (CBRet - CBPerm)
    dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + Fpg) / VS) * (CCRet - CCPerm)

    K_A_diff=CAPerm-K_A*CARet
    K_B_diff=CBPerm-K_B*CBRet
    K_C_diff=CCPerm-K_C*CCRet

    output=[dCA1_dt,dCB1_dt,dCC1_dt,dCARet_dt,dCBRet_dt,dCCRet_dt,K_A_diff,K_B_diff,K_C_diff]
    return output

def ss_DM_rK_A_b(p):
    # CA1, CB1, CC1, CCPerm, r1, r2, K_A, K_B, K_C = y
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    VS=p['VS']
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']

    CAPerm = p['CAPerm']
    CBPerm = p['CBPerm']
    Fpg=p['Fpg']

    # dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    # dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    # dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    # dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + Fpg) / VS) * (CARet - CAPerm)
    # dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + Fpg) / VS) * (CBRet - CBPerm)
    # dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + Fpg) / VS) * (CCRet - CCPerm)

    # K_A_diff=CAPerm-K_A*CARet
    # K_B_diff=CBPerm-K_B*CBRet
    # K_C_diff=CCPerm-K_C*CCRet
    b=np.array([F0/VR*CA0+FR/VR*CAPerm,-FR/VR*CBPerm,0,(FR+Fpg-F1)/VS*CARet-(FR+Fpg)/VS*CAPerm,(FR+Fpg-F1)/VS*CBRet-(FR+Fpg)/VS*CBPerm,
                (FR+Fpg-F1)/VS*CCRet,CAPerm,CBPerm,0])
    # CA1, CB1, CC1, CCPerm, r1, r2, K_A, K_B, K_C
    A=np.array([
        [(F0+FR)/VR,0,0,0,1,0,0,0,0],
        [0,-(F0+FR)/VR,0,0,1,-1,0,0,0],
        [0,0,(F0+FR)/VR,-FR/VR,0,-1,0,0,0],
        [-F1/VS,0,0,0,0,0,0,0,0],
        [0,-F1/VS,0,0,0,0,0,0,0],
        [0,0,-F1/VS,(FR+Fpg)/VS,0,0,0,0,0],
        [0,0,0,0,0,0,CARet,0,0],
        [0,0,0,0,0,0,0,CBRet,0],
        [0,0,0,-1,0,0,0,0,CCRet]    
    ])

    return A,b

# solve for the reaction/partition unknowns row by row
# unknown_names_rK = ['CC1', 'CAPerm', 'CBPerm', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C']
unknown_names_rK = ['CA1','CB1','CC1', 'CCPerm', 'r1', 'r2', 'K_A', 'K_B', 'K_C']
results_rK = []
for i, row in data_copy_3.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    # initial guess: noisy row values for states, rough magnitudes for constitutive terms
    # y0 = [row['CC1'], row['CAPerm'], row['CBPerm'], row['CCPerm'],
    #       0.1, 0.1, 0.5, 0.5, 0.5]
    # y0 = [row['CA1'], row['CB1'], row['CC1'], row['CCPerm'],
    #           0.1, 0.1, 0.5, 0.5, 0.5]
    A, b = ss_DM_rK_A_b(p)
        # 
    Dr, Dc = ruiz_scaling(A)
    A_scaled = Dr[:, None] * A * Dc[None, :]
    sol = Dc * np.linalg.solve(A_scaled, Dr * b)
    sol_unscaled = np.linalg.solve(A, b)
    # res = dict(zip(unknown_names_rK, sol))
    res = dict(zip(unknown_names_rK, sol))
    res['condition number']=np.linalg.cond(A)
    res['condition number scaled']=np.linalg.cond(A_scaled)
    res['max rel diff scaled vs unscaled']=np.max(np.abs(sol-sol_unscaled)/np.abs(sol_unscaled))
    res['rank']=np.linalg.matrix_rank(A)
    # sol = fsolve(ss_DM_rK, y0, args=(p,))

    
    # res['residual_norm'] = np.linalg.norm(ss_DM_rK(sol, p))
    # res['converged'] = res['residual_norm'] < 1e-6
    results_rK.append(res)

results_rK = pd.DataFrame(results_rK, index=data_copy_3.index)
# measured columns are kept from data_copy_3, unknown columns are replaced with solved values
# actually retrieves r1 and r2 well
data_DM_rK = data_copy_3.drop(columns=[c for c in unknown_names_rK if c in data_copy_3.columns]).join(results_rK)
# print(data_DM_rK[unknown_names_rK + ['converged', 'residual_norm']].to_string())
# print(f"Converged: {data_DM_rK['converged'].sum()}/{len(data_DM_rK)}, max residual norm: {data_DM_rK['residual_norm'].max():.3e}")
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


def ss_DM_heat_plus_rK(p, sigma):
    # sigma: dict of measurement noise std devs, used to weight the overdetermined least squares
    # Measured: T1, TjR, TS, CARet, CBRet, CCRet, CAPerm, CBPerm
    # overdet vars CA1, CB1, r1, r2 theta_QR
    # Welldet vars TjS, CC1, CCPerm K_A, K_B, K_C, theta_QS
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    FjR=p['FjR']
    FH=p['FH']
    # CC1=p['CC1']
    T1=p['T1']
    TS=p['TS']
    TjR=p['TjR']
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']
    CAPerm=p['CAPerm']
    CBPerm=p['CBPerm']

    T0=p['T0']
    Fpg=p['Fpg']
    VjR=p['VjR']
    TCin=p['TCin']
    VjS=p['VjS']
    THin=p['THin']
    dH1=p['dH1']
    dH2=p['dH2']
    rho=p['rho']
    Cp=p['Cp']
    rhoc=p['rhoc']
    Cpc=p['Cpc']
    rhoH=p['rhoH']
    CpH=p['CpH']

    # solve a least squares problem with the overdetermined (CA1, CB1, r1, r2 theta_QR) and the following equations:
    # dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    # dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    # dT1_dt = ((F0 * (p['T0'] - T1) + FR * (TS - T1)) / VR
    #               + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp'])
    #               - theta_QR / (p['rho'] * p['Cp'] * VR))
    # dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])
    # dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + Fpg) / VS) * (CARet - CAPerm)
    # dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + Fpg) / VS) * (CBRet - CBPerm)
    # CA1, CB1, r1, r2, theta_QR
    b_od=np.array([F0/VR*CA0+FR/VR*CAPerm,-FR/VR*CBPerm,(F0*(T0-T1)+FR*(TS-T1))/VR,FjR/VjR*(TCin-TjR),
                   (FR+Fpg-F1)/VS*CARet-(FR+Fpg)/VS*CAPerm,(FR+Fpg-F1)/VS*CBRet-(FR+Fpg)/VS*CBPerm])
    A_od=np.array([
        [(F0+FR)/VR,0,1,0,0],
        [0,-(F0+FR)/VR,1,-1,0],
        [0,0,dH1/(rho*Cp),dH2/(rho*Cp),1/(rho*Cp*VR)],
        [0,0,0,0,-1/(rhoc*Cpc*VjR)],
        [-F1/VS,0,0,0,0],
        [0,-F1/VS,0,0,0]
    ])
    # weight each equation by the noise it carries (generalized least squares)
    # A_od has no measurements in it, so the residual noise is J * (measurement noise), J = db_od/dm
    # measurements m: T1, TjR, TS, CARet, CBRet, CAPerm, CBPerm
    meas_od = ['T1', 'TjR', 'TS', 'CARet', 'CBRet', 'CAPerm', 'CBPerm']
    J_od=np.array([
        [0,0,0,0,0,FR/VR,0],
        [0,0,0,0,0,0,-FR/VR],
        [-(F0+FR)/VR,0,FR/VR,0,0,0,0],
        [0,-FjR/VjR,0,0,0,0,0],
        [0,0,0,(FR+Fpg-F1)/VS,0,-(FR+Fpg)/VS,0],
        [0,0,0,0,(FR+Fpg-F1)/VS,0,-(FR+Fpg)/VS]
    ])
    sig_od = np.array([sigma[m] for m in meas_od])
    # residual covariance (equations sharing a measurement have correlated errors)
    Sigma_od = (J_od * sig_od**2) @ J_od.T
    L_od = np.linalg.cholesky(Sigma_od)
    # whiten: min ||L^-1 (A x - b)||
    A_od_w = np.linalg.solve(L_od, A_od)
    b_od_w = np.linalg.solve(L_od, b_od)
    # column scaling only (does not change the solution, just the conditioning), x = Dc z
    _, Dc_od = ruiz_scaling(A_od_w)
    z_od = np.linalg.lstsq(A_od_w * Dc_od[None, :], b_od_w, rcond=None)[0]
    sol_od = Dc_od * z_od
    CA1, CB1, r1, r2, theta_QR = sol_od

    # then use the results of the least squares solution to compute the well-det vars (TjS, CC1, CCPerm K_A, K_B, K_C, theta_QS) using the following equations:
    # dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    # dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + Fpg) / VS) * (CCRet - CCPerm)
    # dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    # dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    # K_A_diff=CAPerm-K_A*CARet
    # K_B_diff=CBPerm-K_B*CBRet
    # K_C_diff=CCPerm-K_C*CCRet
    # TjS, CC1, CCPerm, K_A, K_B, K_C, theta_QS
    b_wd=np.array([r2,(FR+Fpg-F1)/VS*CCRet,F1/VS*(T1-TS),FH/VjS*THin,CAPerm,CBPerm,0])
    A_wd=np.array([
        [0,(F0+FR)/VR,-FR/VR,0,0,0,0],
        [0,-F1/VS,(FR+Fpg)/VS,0,0,0,0],
        [0,0,0,0,0,0,-1/(rho*Cp*VS)],
        [FH/VjS,0,0,0,0,0,1/(rhoH*CpH*VjS)],
        [0,0,0,CARet,0,0,0],
        [0,0,0,0,CBRet,0,0],
        [0,0,-1,0,0,CCRet,0]
    ])
    Dr_wd, Dc_wd = ruiz_scaling(A_wd)
    A_wd_scaled = Dr_wd[:, None] * A_wd * Dc_wd[None, :]
    sol_wd = Dc_wd * np.linalg.solve(A_wd_scaled, Dr_wd * b_wd)

    # the function outputs the solutions to both the overdetermined and well-determined values
    return sol_od, sol_wd

# solve for the heat + reaction/partition unknowns row by row
data_copy_4=data_copy.copy()
unknown_names_od = ['CA1', 'CB1', 'r1', 'r2', 'theta_QR']
unknown_names_wd = ['TjS', 'CC1', 'CCPerm', 'K_A', 'K_B', 'K_C', 'theta_QS']
unknown_names_heat_rK = unknown_names_od + unknown_names_wd
results_heat_rK = []
for i, row in data_copy_4.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    sol_od, sol_wd = ss_DM_heat_plus_rK(p, noise_sigma_DM)
    res = dict(zip(unknown_names_od, sol_od))
    res.update(zip(unknown_names_wd, sol_wd))
    results_heat_rK.append(res)

results_heat_rK = pd.DataFrame(results_heat_rK, index=data_copy_4.index)
# measured columns are kept from data_copy_4, unknown columns are replaced with solved values
data_DM_heat_rK = data_copy_4.drop(columns=[c for c in unknown_names_heat_rK if c in data_copy_4.columns]).join(results_heat_rK)
data_DM_heat_rK.to_csv('noisy_heat_r_kappa_DM.csv', index=False)

# make parity plots of the solved variables with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_heat_rK
parity_vars_heat_rK = unknown_names_heat_rK
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
for ax, var in zip(axes.ravel(), parity_vars_heat_rK):
    x = data_clean[var].to_numpy(dtype=float)
    y = data_DM_heat_rK[var].to_numpy(dtype=float)
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
plt.savefig('noisy_heat_r_kappa_DM_parity.png', dpi=150)
plt.close(fig)
print("Saved noisy_heat_r_kappa_DM_parity.png")

# evaluate the 
def ss_DM_heat_plus_rK_plus_CCPerm(p, sigma):
    # Measured: T1, TjR, TS, CARet, CBRet, CCRet, CAPerm, CBPerm, CCPerm
    # overdet vars CA1, CB1, CC1, r1, r2 theta_QR
    # Welldet vars TjS K_A, K_B, K_C, theta_QS
    # sigma: dict of measurement noise std devs, used to weight the overdetermined least squares
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    FjR=p['FjR']
    FH=p['FH']
    T1=p['T1']
    TS=p['TS']
    TjR=p['TjR']
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']
    CAPerm=p['CAPerm']
    CBPerm=p['CBPerm']
    CCPerm=p['CCPerm']

    T0=p['T0']
    Fpg=p['Fpg']
    VjR=p['VjR']
    TCin=p['TCin']
    VjS=p['VjS']
    THin=p['THin']
    dH1=p['dH1']
    dH2=p['dH2']
    rho=p['rho']
    Cp=p['Cp']
    rhoc=p['rhoc']
    Cpc=p['Cpc']
    rhoH=p['rhoH']
    CpH=p['CpH']

    # solve a least squares problem with the overdetermined (CA1, CB1, CC1, r1, r2 theta_QR) and the following equations:
    # dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    # dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    # dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    # dT1_dt = ((F0 * (p['T0'] - T1) + FR * (TS - T1)) / VR
    #             + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp'])
    #             - theta_QR / (p['rho'] * p['Cp'] * VR))
    # dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])
    # dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    # dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    # dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    # CA1, CB1, CC1, r1, r2, theta_QR
    b_od=np.array([F0/VR*CA0+FR/VR*CAPerm,-FR/VR*CBPerm,FR/VR*CCPerm,(F0*(T0-T1)+FR*(TS-T1))/VR,FjR/VjR*(TCin-TjR),
                   (FR+Fpg-F1)/VS*CARet-(FR+Fpg)/VS*CAPerm,(FR+Fpg-F1)/VS*CBRet-(FR+Fpg)/VS*CBPerm,
                   (FR+Fpg-F1)/VS*CCRet-(FR+Fpg)/VS*CCPerm])
    A_od=np.array([
        [(F0+FR)/VR,0,0,1,0,0],
        [0,-(F0+FR)/VR,0,1,-1,0],
        [0,0,(F0+FR)/VR,0,-1,0],
        [0,0,0,dH1/(rho*Cp),dH2/(rho*Cp),1/(rho*Cp*VR)],
        [0,0,0,0,0,-1/(rhoc*Cpc*VjR)],
        [-F1/VS,0,0,0,0,0],
        [0,-F1/VS,0,0,0,0],
        [0,0,-F1/VS,0,0,0]
    ])
    # weight each equation by the noise it carries (generalized least squares)
    # A_od has no measurements in it, so the residual noise is J * (measurement noise), J = db_od/dm
    # measurements m: T1, TjR, TS, CARet, CBRet, CCRet, CAPerm, CBPerm, CCPerm
    meas_od = ['T1', 'TjR', 'TS', 'CARet', 'CBRet', 'CCRet', 'CAPerm', 'CBPerm', 'CCPerm']
    J_od=np.array([
        [0,0,0,0,0,0,FR/VR,0,0],
        [0,0,0,0,0,0,0,-FR/VR,0],
        [0,0,0,0,0,0,0,0,FR/VR],
        [-(F0+FR)/VR,0,FR/VR,0,0,0,0,0,0],
        [0,-FjR/VjR,0,0,0,0,0,0,0],
        [0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS,0,0],
        [0,0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS,0],
        [0,0,0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS]
    ])
    sig_od = np.array([sigma[m] for m in meas_od])
    # residual covariance (equations sharing a measurement have correlated errors)
    Sigma_od = (J_od * sig_od**2) @ J_od.T
    L_od = np.linalg.cholesky(Sigma_od)
    # whiten: min ||L^-1 (A x - b)||
    A_od_w = np.linalg.solve(L_od, A_od)
    b_od_w = np.linalg.solve(L_od, b_od)
    # column scaling only (does not change the solution, just the conditioning), x = Dc z
    _, Dc_od = ruiz_scaling(A_od_w)
    z_od = np.linalg.lstsq(A_od_w * Dc_od[None, :], b_od_w, rcond=None)[0]
    sol_od = Dc_od * z_od

    # then use the results of the least squares solution to compute the well-det vars (TjS, K_A, K_B, K_C, theta_QS) using the following equations:
    # dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    # dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    # K_A_diff=CAPerm-K_A*CARet
    # K_B_diff=CBPerm-K_B*CBRet
    # K_C_diff=CCPerm-K_C*CCRet
    # note: none of these depend on the least squares solution, since CCPerm is now measured
    # TjS, K_A, K_B, K_C, theta_QS
    b_wd=np.array([F1/VS*(T1-TS),FH/VjS*THin,CAPerm,CBPerm,CCPerm])
    A_wd=np.array([
        [0,0,0,0,-1/(rho*Cp*VS)],
        [FH/VjS,0,0,0,1/(rhoH*CpH*VjS)],
        [0,CARet,0,0,0],
        [0,0,CBRet,0,0],
        [0,0,0,CCRet,0]
    ])
    Dr_wd, Dc_wd = ruiz_scaling(A_wd)
    A_wd_scaled = Dr_wd[:, None] * A_wd * Dc_wd[None, :]
    sol_wd = Dc_wd * np.linalg.solve(A_wd_scaled, Dr_wd * b_wd)

    # the function outputs the solutions to both the overdetermined and well-determined values
    return sol_od, sol_wd

# solve for the heat + reaction/partition unknowns (with CCPerm measured) row by row
data_copy_5=data_copy.copy()
unknown_names_od_CCPerm = ['CA1', 'CB1', 'CC1', 'r1', 'r2', 'theta_QR']
unknown_names_wd_CCPerm = ['TjS', 'K_A', 'K_B', 'K_C', 'theta_QS']
unknown_names_heat_rK_CCPerm = unknown_names_od_CCPerm + unknown_names_wd_CCPerm
results_heat_rK_CCPerm = []
for i, row in data_copy_5.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    sol_od, sol_wd = ss_DM_heat_plus_rK_plus_CCPerm(p, noise_sigma_DM)
    res = dict(zip(unknown_names_od_CCPerm, sol_od))
    res.update(zip(unknown_names_wd_CCPerm, sol_wd))
    results_heat_rK_CCPerm.append(res)

results_heat_rK_CCPerm = pd.DataFrame(results_heat_rK_CCPerm, index=data_copy_5.index)
# measured columns are kept from data_copy_5, unknown columns are replaced with solved values
data_DM_heat_rK_CCPerm = data_copy_5.drop(columns=[c for c in unknown_names_heat_rK_CCPerm if c in data_copy_5.columns]).join(results_heat_rK_CCPerm)
data_DM_heat_rK_CCPerm.to_csv('noisy_heat_r_kappa_CCPerm_DM.csv', index=False)

# make parity plots of the solved variables (plus measured CCPerm) with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_heat_rK_CCPerm
parity_vars_heat_rK_CCPerm = unknown_names_heat_rK_CCPerm + ['CCPerm']
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
for ax, var in zip(axes.ravel(), parity_vars_heat_rK_CCPerm):
    x = data_clean[var].to_numpy(dtype=float)
    y = data_DM_heat_rK_CCPerm[var].to_numpy(dtype=float)
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
plt.savefig('noisy_heat_r_kappa_CCPerm_DM_parity.png', dpi=150)
plt.close(fig)
print("Saved noisy_heat_r_kappa_CCPerm_DM_parity.png")


def ss_DM_heat_plus_rK_plus_CCPerm_plus_TjS(p, sigma):
    # Measured: T1, TjR, TS, TjS, CARet, CBRet, CCRet, CAPerm, CBPerm, CCPerm
    # overdet vars CA1, CB1, CC1, r1, r2, theta_QR, theta_QS
    # Welldet vars K_A, K_B, K_C
    # sigma: dict of measurement noise std devs, used to weight the overdetermined least squares
    VR=p['VR']
    VS=p['VS']
    F0=p['F0']
    CA0=p['CA0']
    FR=p['FR']
    F1=p['F1']
    FjR=p['FjR']
    FH=p['FH']
    T1=p['T1']
    TS=p['TS']
    TjR=p['TjR']
    TjS=p['TjS']
    CARet=p['CARet']
    CBRet=p['CBRet']
    CCRet=p['CCRet']
    CAPerm=p['CAPerm']
    CBPerm=p['CBPerm']
    CCPerm=p['CCPerm']

    T0=p['T0']
    Fpg=p['Fpg']
    VjR=p['VjR']
    TCin=p['TCin']
    VjS=p['VjS']
    THin=p['THin']
    dH1=p['dH1']
    dH2=p['dH2']
    rho=p['rho']
    Cp=p['Cp']
    rhoc=p['rhoc']
    Cpc=p['Cpc']
    rhoH=p['rhoH']
    CpH=p['CpH']
    # solve a least squares problem with the overdetermined (CA1, CB1, CC1, r1, r2 theta_QR, theta_QS) and the following equations:
    # dCA1_dt = (F0 / VR) * (CA0 - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    # dCB1_dt = -(F0 / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    # dCC1_dt = -(F0 / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    # dT1_dt = ((F0 * (p['T0'] - T1) + FR * (TS - T1)) / VR
    #             + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp'])
    #             - theta_QR / (p['rho'] * p['Cp'] * VR))
    # dTjR_dt = (FjR / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])
    # dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    # dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    # dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    # dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    # dTjS_dt = (FH / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    # CA1, CB1, CC1, r1, r2, theta_QR, theta_QS
    b_od=np.array([F0/VR*CA0+FR/VR*CAPerm,-FR/VR*CBPerm,FR/VR*CCPerm,(F0*(T0-T1)+FR*(TS-T1))/VR,FjR/VjR*(TCin-TjR),
                   (FR+Fpg-F1)/VS*CARet-(FR+Fpg)/VS*CAPerm,(FR+Fpg-F1)/VS*CBRet-(FR+Fpg)/VS*CBPerm,
                   (FR+Fpg-F1)/VS*CCRet-(FR+Fpg)/VS*CCPerm,F1/VS*(T1-TS),FH/VjS*(THin-TjS)])
    A_od=np.array([
        [(F0+FR)/VR,0,0,1,0,0,0],
        [0,-(F0+FR)/VR,0,1,-1,0,0],
        [0,0,(F0+FR)/VR,0,-1,0,0],
        [0,0,0,dH1/(rho*Cp),dH2/(rho*Cp),1/(rho*Cp*VR),0],
        [0,0,0,0,0,-1/(rhoc*Cpc*VjR),0],
        [-F1/VS,0,0,0,0,0,0],
        [0,-F1/VS,0,0,0,0,0],
        [0,0,-F1/VS,0,0,0,0],
        [0,0,0,0,0,0,-1/(rho*Cp*VS)],
        [0,0,0,0,0,0,1/(rhoH*CpH*VjS)]
    ])
    # weight each equation by the noise it carries (generalized least squares)
    # A_od has no measurements in it, so the residual noise is J * (measurement noise), J = db_od/dm
    # measurements m: T1, TjR, TS, TjS, CARet, CBRet, CCRet, CAPerm, CBPerm, CCPerm
    meas_od = ['T1', 'TjR', 'TS', 'TjS', 'CARet', 'CBRet', 'CCRet', 'CAPerm', 'CBPerm', 'CCPerm']
    J_od=np.array([
        [0,0,0,0,0,0,0,FR/VR,0,0],
        [0,0,0,0,0,0,0,0,-FR/VR,0],
        [0,0,0,0,0,0,0,0,0,FR/VR],
        [-(F0+FR)/VR,0,FR/VR,0,0,0,0,0,0,0],
        [0,-FjR/VjR,0,0,0,0,0,0,0,0],
        [0,0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS,0,0],
        [0,0,0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS,0],
        [0,0,0,0,0,0,(FR+Fpg-F1)/VS,0,0,-(FR+Fpg)/VS],
        [F1/VS,0,-F1/VS,0,0,0,0,0,0,0],
        [0,0,0,-FH/VjS,0,0,0,0,0,0]
    ])
    sig_od = np.array([sigma[m] for m in meas_od])
    # residual covariance (equations sharing a measurement have correlated errors, e.g. the T1 and TS balances)
    Sigma_od = (J_od * sig_od**2) @ J_od.T
    L_od = np.linalg.cholesky(Sigma_od)
    # whiten: min ||L^-1 (A x - b)||
    A_od_w = np.linalg.solve(L_od, A_od)
    b_od_w = np.linalg.solve(L_od, b_od)
    # column scaling only (does not change the solution, just the conditioning), x = Dc z
    _, Dc_od = ruiz_scaling(A_od_w)
    z_od = np.linalg.lstsq(A_od_w * Dc_od[None, :], b_od_w, rcond=None)[0]
    sol_od = Dc_od * z_od

    # then use the results of the least squares solution to compute the well-det vars (K_A, K_B, K_C) using the following equations:
    # K_A_diff=CAPerm-K_A*CARet
    # K_B_diff=CBPerm-K_B*CBRet
    # K_C_diff=CCPerm-K_C*CCRet
    # note: these only involve measured values, so they do not depend on the least squares solution
    # K_A, K_B, K_C
    b_wd=np.array([CAPerm,CBPerm,CCPerm])
    A_wd=np.diag([CARet,CBRet,CCRet])
    sol_wd = np.linalg.solve(A_wd, b_wd)

    # the function outputs the solutions to both the overdetermined and well-determined values
    return sol_od, sol_wd

# solve for the heat + reaction/partition unknowns (with CCPerm and TjS measured) row by row
data_copy_6=data_copy.copy()
unknown_names_od_CCPerm_TjS = ['CA1', 'CB1', 'CC1', 'r1', 'r2', 'theta_QR', 'theta_QS']
unknown_names_wd_CCPerm_TjS = ['K_A', 'K_B', 'K_C']
unknown_names_heat_rK_CCPerm_TjS = unknown_names_od_CCPerm_TjS + unknown_names_wd_CCPerm_TjS
results_heat_rK_CCPerm_TjS = []
for i, row in data_copy_6.iterrows():
    p = {**params, **row.to_dict()}  # measured values for this row override defaults (F0, FjR, FH)
    sol_od, sol_wd = ss_DM_heat_plus_rK_plus_CCPerm_plus_TjS(p, noise_sigma_DM)
    res = dict(zip(unknown_names_od_CCPerm_TjS, sol_od))
    res.update(zip(unknown_names_wd_CCPerm_TjS, sol_wd))
    results_heat_rK_CCPerm_TjS.append(res)

results_heat_rK_CCPerm_TjS = pd.DataFrame(results_heat_rK_CCPerm_TjS, index=data_copy_6.index)
# measured columns are kept from data_copy_6, unknown columns are replaced with solved values
data_DM_heat_rK_CCPerm_TjS = data_copy_6.drop(columns=[c for c in unknown_names_heat_rK_CCPerm_TjS if c in data_copy_6.columns]).join(results_heat_rK_CCPerm_TjS)
data_DM_heat_rK_CCPerm_TjS.to_csv('noisy_heat_r_kappa_CCPerm_TjS_DM.csv', index=False)

# make parity plots of the solved variables (plus measured CCPerm and TjS) with x axes being non-noisy from simulation and y_axes being their corresponding values in data_DM_heat_rK_CCPerm_TjS
parity_vars_heat_rK_CCPerm_TjS = unknown_names_heat_rK_CCPerm_TjS + ['CCPerm', 'TjS']
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
for ax, var in zip(axes.ravel(), parity_vars_heat_rK_CCPerm_TjS):
    x = data_clean[var].to_numpy(dtype=float)
    y = data_DM_heat_rK_CCPerm_TjS[var].to_numpy(dtype=float)
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
plt.savefig('noisy_heat_r_kappa_CCPerm_TjS_DM_parity.png', dpi=150)
plt.close(fig)
print("Saved noisy_heat_r_kappa_CCPerm_TjS_DM_parity.png")
