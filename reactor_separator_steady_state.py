import numpy as np
from scipy.optimize import fsolve
import pandas as pd

# =============================================================================
# 1. PARAMETERS & SPECIFIED CONSTANTS
# =============================================================================
params = {
    # Operating & Feed Conditions """Can change F0, CA0, T0 when testing different scenarios. Fpg and Fperm are fixed"""
    
    'F0': 10.0,          # Feed flow rate [m3/h]
    'CA0': 1.0,          # Feed concentration of component A [kmol/m3]
    'T0': 360.0,         # Feed temperature [K]

    'Fpg': 0.2,          # Purge flow rate [m3/h]
    'Fperm': 5.5,        # Permeate flow rate [m3/h]
    
    # Fluid Properties """These should not be changed"""
    
    'rho': 1000.0,       # Reaction fluid density [kg/m3]
    'Cp': 4.184,         # Reaction fluid heat capacity [kJ/(kg·K)]
    'rhoc': 1000.0,      # Cooling fluid density [kg/m3]
    'Cpc': 4.184,        # Cooling fluid heat capacity [kJ/(kg·K)]
    'rhoH': 1000.0,      # Heating fluid density [kg/m3]
    'CpH': 4.184,        # Heating fluid heat capacity [kJ/(kg·K)]
    
    # Reaction Enthalpies & Kinetics """These should not be changed"""
    
    'dH1': -50000.0,     # Enthalpy of reaction 1 [kJ/kmol] (exothermic)
    'dH2': -30000.0,     # Enthalpy of reaction 2 [kJ/kmol] (exothermic)
    'A1': 1e6,           # Arrhenius pre-exponential factor 1 [1/h]
    'EA1': 40000.0,      # Activation energy 1 [J/mol]
    'A2': 5e5,           # Arrhenius pre-exponential factor 2 [1/h]
    'EA2': 35000.0,      # Activation energy 2 [J/mol]
    'R': 8.314,          # Universal gas constant [J/(mol·K)]
    
    # Separation & Permeation Properties """These should not be changed"""
    
    'dGA': 1000.0,       # Gibbs energy parameter for A [J/mol]
    'dGB': 2000.0,       # Gibbs energy parameter for B [J/mol]
    'dGC': 3000.0,       # Gibbs energy parameter for C [J/mol]
    
    # Heat Transfer & Jackets """FjR, TCin, FH, THin can be changed to test different scenarios. VjR, VjS, UaR, UaS should not be changed"""
    
    'VjR': 2.0,          # Reactor jacket volume [m3]
    'FjR': 20.0,         # Cooling fluid flow rate [m3/h]
    'TCin': 290.0,       # Cooling fluid inlet temperature [K]
    'VjS': 1.5,          # Separator jacket volume [m3]
    'FH': 10.0,          # Heating fluid flow rate [m3/h]
    'THin': 400.0,       # Heating fluid inlet temperature [K]
    'UaR': 5000.0,       # Volumetric heat transfer coeff. (reactor) [kJ/(m3·h·K)]
    'UaS': 4000.0,       # Volumetric heat transfer coeff. (separator) [kJ/(m3·h·K)]
    
    # Hydraulic Valve Constants """These should not be changed"""
    
    'kvR': 2.5,          # Reactor discharge valve constant [m3/h / sqrt(m3)]
    'kvS': 2.0           # Separator discharge valve constant [m3/h / sqrt(m3)]
}

# =============================================================================
# 2. STEADY-STATE ALGEBRAIC SYSTEM (Derivatives = 0)
# =============================================================================
def steady_state_system(y, p):
    """
    Returns the derivatives (which we want to be exactly 0).
    Time 't' is removed from the signature.
    """
    VR, CA1, CB1, CC1, T1, TjR, VS, CARet, CBRet, CCRet, TS, TjS = y

    # --- Algebraic Equations ---
    F1 = p['kvR'] * np.sqrt(max(VR, 1e-6))
    FRet = p['kvS'] * np.sqrt(max(VS, 1e-6))
    FR = p['Fperm'] - p['Fpg']

    # Partition coefficients
    kappa_A = np.exp(-p['dGA'] / (p['R'] * TS))
    kappa_B = np.exp(-p['dGB'] / (p['R'] * TS))
    kappa_C = np.exp(-p['dGC'] / (p['R'] * TS))

    # Permeate concentrations
    CAPerm = kappa_A * CARet
    CBPerm = kappa_B * CBRet
    CCPerm = kappa_C * CCRet

    # Reaction rates
    r1 = p['A1'] * np.exp(-p['EA1'] / (p['R'] * T1)) * CA1
    r2 = p['A2'] * np.exp(-p['EA2'] / (p['R'] * T1)) * CB1

    # Heat transfer rates
    theta_QR = p['UaR'] * VR * (T1 - TjR)
    theta_QS = p['UaS'] * VS * (TjS - TS)

    # --- Steady State Balances (d(Var)/dt = 0) ---
    # Reactor balance
    dVR_dt = p['F0'] + FR - F1
    dCA1_dt = (p['F0'] / VR) * (p['CA0'] - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    dCB1_dt = -(p['F0'] / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    dCC1_dt = -(p['F0'] / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    dT1_dt = ((p['F0'] * (p['T0'] - T1) + FR * (TS - T1)) / VR 
              + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp']) 
              - theta_QR / (p['rho'] * p['Cp'] * VR))
    dTjR_dt = (p['FjR'] / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])

    # Separator balance
    dVS_dt = F1 - FRet - FR - p['Fpg']
    dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    dTjS_dt = (p['FH'] / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])

    return [dVR_dt, dCA1_dt, dCB1_dt, dCC1_dt, dT1_dt, dTjR_dt,
            dVS_dt, dCARet_dt, dCBRet_dt, dCCRet_dt, dTS_dt, dTjS_dt]

# =============================================================================
# 3. ROOT FINDING (SOLVING FOR STEADY STATE)
# =============================================================================
# Initial guess for the roots. 
# Using small non-zero values for concentrations avoids potential division/multiplication by zero issues during solver iterations.
y0_guess = [
    10.0,             # VR_guess [m3]
    0.5,              # CA1_guess [kmol/m3]
    0.1,              # CB1_guess [kmol/m3]
    0.1,              # CC1_guess [kmol/m3]
    340.0,            # T1_guess [K]
    300.0,            # TjR_guess [K]
    5.0,              # VS_guess [m3]
    0.5,              # CARet_guess [kmol/m3]
    0.1,              # CBRet_guess [kmol/m3]
    0.1,              # CCRet_guess [kmol/m3]
    params['TCin'],   # TS_guess [K]
    params['THin']    # TjS_guess [K]
]

# Solve the nonlinear system
solution, infodict, ier, mesg = fsolve(steady_state_system, y0_guess, args=(params,), full_output=True)

# =============================================================================
# 4. OUTPUT RESULTS
# =============================================================================
if ier == 1:
    print("Steady-state successfully found!\n")
    state_names = [
        "Reactor Volume (VR)", "Reactor Conc. A (CA1)", "Reactor Conc. B (CB1)", "Reactor Conc. C (CC1)", 
        "Reactor Temp (T1)", "Reactor Jacket Temp (TjR)", "Separator Volume (VS)", 
        "Retentate Conc. A (CARet)", "Retentate Conc. B (CBRet)", "Retentate Conc. C (CCRet)", 
        "Separator Temp (TS)", "Separator Jacket Temp (TjS)"
    ]
    
    units = [
        "[m^3]", "[kmol/m^3]", "[kmol/m^3]", "[kmol/m^3]", 
        "[K]", "[K]", "[m^3]", 
        "[kmol/m^3]", "[kmol/m^3]", "[kmol/m^3]", 
        "[K]", "[K]"
    ]
    
    print("-" * 50)
    print(f"{'State Variable':<30} | {'Value':>10} | {'Unit'}")
    print("-" * 50)
    for name, val, unit in zip(state_names, solution, units):
        print(f"{name:<30} | {val:10.4f} | {unit}")
    print("-" * 50)
else:
    print("Solver failed to converge.")
    print("Message:", mesg)


# I would like to generate training data to learn the heat transfer coefficients
# vary the F0 that goes into the reactor as this will affect the volumes of the tanks
# for every F0, vary the inlet flow of the cooling fluid and the heating fluid to see how the heat transfer coefficients change
# store the solver results for each combination of F0, TCin, and THin to create a dataset for training a model to predict heat transfer coefficients based on these inputs.
#  also add F1, FR, and FRet to the dataset
df = pd.DataFrame(columns=[
    'F0', 'FjR', 'FH',
    'VR', 'CA1', 'CB1', 'CC1', 'T1', 'TjR', 'VS', 'CARet', 'CBRet', 'CCRet', 'TS', 'TjS','CAPerm', 'CBPerm', 'CCPerm','F1', 'FR', 'FRet','r1','r2','K_A','K_B','K_C','theta_QR','theta_QS'])
for F0 in np.linspace(5, 10, 5):  # Varying F0 from 5 to 10 m3/h
    for FjR in np.linspace(5, 20, 5):  # Varying cooling fluid inlet flow from 15 to 45 m3/h
        for FH in np.linspace(5, 20, 5):  # Varying heating fluid inlet flow from 5 to 35 m3/h
            # Update parameters
            params['F0'] = F0
            params['FjR'] = FjR
            params['FH'] = FH

            # Solve the nonlinear system for the current set of parameters
            solution, infodict, ier, mesg = fsolve(steady_state_system, y0_guess, args=(params,), full_output=True)
            
            if ier == 1:
                # Extract relevant outputs for training data
                VR, CA1, CB1, CC1, T1, TjR, VS, CARet, CBRet, CCRet, TS, TjS = solution
                # get Caperm, CBperm, CCperm
                kappa_A = np.exp(-params['dGA'] / (params['R'] * TS))
                kappa_B = np.exp(-params['dGB'] / (params['R'] * TS))
                kappa_C = np.exp(-params['dGC'] / (params['R'] * TS))
                CAPerm = kappa_A * CARet
                CBPerm = kappa_B * CBRet
                CCPerm = kappa_C * CCRet
                # get F1, FR, FRet
                F1 = params['kvR'] * np.sqrt(max(VR, 1e-6))
                FRet = params['kvS'] * np.sqrt(max(VS, 1e-6))
                FR = params['Fperm'] - params['Fpg']

                # add calculated constitutive terms to the DF, compute the rxn rates and heat transfer terms
                theta_QR=params['UaR']*VR*(T1-TjR)
                theta_QS=params['UaS']*VS*(TjS-TS)
                r1=params['A1']*np.exp(-params['EA1'] / (params['R'] * T1))*CA1
                r2=params['A2']*np.exp(-params['EA2'] / (params['R'] * T1))*CB1

                # Store or print the results as needed for training data adding the permeate concentrations to the dataframe
                print(f"F0: {F0:.2f}, FjR: {FjR:.2f}, FH: {FH:.2f} => VR: {VR:.4f}, VS: {VS:.4f}, TjR: {TjR:.4f}, T1: {T1:.4f},TjS: {TjS:.4f}, TS: {TS:.4f}")
                # Append the results to the DataFrame
                df.loc[len(df)] = {'F0': F0, 'FjR': FjR, 'FH': FH,'VR': VR, 'CA1': CA1, 
                                   'CB1': CB1, 'CC1': CC1, 'T1': T1, 'TjR': TjR,'VS': VS, 
                                   'CARet': CARet, 'CBRet': CBRet, 'CCRet': CCRet, 'TS': TS, 
                                   'TjS': TjS, 'CAPerm': CAPerm, 'CBPerm': CBPerm, 'CCPerm': CCPerm, 
                                   'F1': F1, 'FR': FR, 'FRet': FRet, 'r1':r1, 'r2':r2,
                                   'K_A':kappa_A,'K_B':kappa_B,'K_C':kappa_C, 
                                   'theta_QR':theta_QR, 'theta_QS':theta_QS}
            else:
                print(f"Solver failed for F0: {F0:.2f}, FjR: {FjR:.2f}, FH: {FH:.2f}. Message: {mesg}")
# Save the DataFrame to a CSV file for training data

# add the other constitutive terms to this df:
df.to_csv('simulator_data_no_noise.csv', index=False)