import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & SPECIFIED CONSTANTS
# =============================================================================
params = {
    # Operating & Feed Conditions
    'F0': 10.0,          # Feed flow rate [m3/h]
    'CA0': 1.0,          # Feed concentration of component A [kmol/m3]
    'T0': 350.0,         # Feed temperature [K]
    'Fpg': 0.5,          # Purge flow rate [m3/h]
    'Fperm': 5.5,        # Permeate flow rate [m3/h]
    
    # Fluid Properties
    'rho': 1000.0,       # Reaction fluid density [kg/m3]
    'Cp': 4.184,         # Reaction fluid heat capacity [kJ/(kg·K)]
    'rhoc': 1000.0,      # Cooling fluid density [kg/m3]
    'Cpc': 4.184,        # Cooling fluid heat capacity [kJ/(kg·K)]
    'rhoH': 1000.0,      # Heating fluid density [kg/m3]
    'CpH': 4.184,        # Heating fluid heat capacity [kJ/(kg·K)]
    
    # Reaction Enthalpies & Kinetics
    'dH1': -50000.0,     # Enthalpy of reaction 1 [kJ/kmol] (exothermic)
    'dH2': -30000.0,     # Enthalpy of reaction 2 [kJ/kmol] (exothermic)
    'A1': 1e6,           # Arrhenius pre-exponential factor 1 [1/h]
    'EA1': 40000.0,      # Activation energy 1 [J/mol]
    'A2': 5e5,           # Arrhenius pre-exponential factor 2 [1/h]
    'EA2': 45000.0,      # Activation energy 2 [J/mol]
    'R': 8.314,          # Universal gas constant [J/(mol·K)]
    
    # Separation & Permeation Properties (0 < dGA < dGB < dGC)
    'dGA': 1000.0,       # Gibbs energy parameter for A [J/mol]
    'dGB': 2000.0,       # Gibbs energy parameter for B [J/mol]
    'dGC': 3000.0,       # Gibbs energy parameter for C [J/mol]
    
    # Heat Transfer & Jackets
    'VjR': 2.0,          # Reactor jacket volume [m3]
    'FjR': 15.0,         # Cooling fluid flow rate [m3/h]
    'TCin': 290.0,       # Cooling fluid inlet temperature [K]
    'VjS': 1.5,          # Separator jacket volume [m3]
    'FH': 10.0,          # Heating fluid flow rate [m3/h]
    'THin': 380.0,       # Heating fluid inlet temperature [K]
    'UaR': 5000.0,       # Volumetric heat transfer coeff. (reactor) [kJ/(m3·h·K)]
    'UaS': 4000.0,       # Volumetric heat transfer coeff. (separator) [kJ/(m3·h·K)]
    
    # Hydraulic Valve Constants
    'kvR': 2.5,          # Reactor discharge valve constant [m3/h / sqrt(m3)]
    'kvS': 2.0           # Separator discharge valve constant [m3/h / sqrt(m3)]
}


# =============================================================================
# 2. DAE SYSTEM FUNCTION (REDUCED TO ODE)
# =============================================================================
def dae_system(t, y, p):
    """
    State vector y:
    y[0]  = VR        : Reactor Volume [m3]
    y[1]  = CA1       : Reactor Concentration A [kmol/m3]
    y[2]  = CB1       : Reactor Concentration B [kmol/m3]
    y[3]  = CC1       : Reactor Concentration C [kmol/m3]
    y[4]  = T1        : Reactor Temperature [K]
    y[5]  = TjR       : Reactor Jacket Temperature [K]
    y[6]  = VS        : Separator Volume [m3]
    y[7]  = CARet     : Retentate Concentration A [kmol/m3]
    y[8]  = CBRet     : Retentate Concentration B [kmol/m3]
    y[9]  = CCRet     : Retentate Concentration C [kmol/m3]
    y[10] = TS        : Separator Temperature [K]
    y[11] = TjS       : Separator Jacket Temperature [K]
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

    # --- Differential Equations ---
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
# 3. INITIAL CONDITIONS & INTEGRATION SETUP
# =============================================================================
# Initial state vector y0
y0 = [
    1.0,   # VR0 [m3]
    0,    # CA1_0 [kmol/m3]
    0,    # CB1_0 [kmol/m3]
    0,   # CC1_0 [kmol/m3]
    340.0,  # T1_0 [K]
    300.0,  # TjR_0 [K]
    1.0,   # VS0 [m3]
    0,    # CARet_0 [kmol/m3]
    0,    # CBRet_0 [kmol/m3]
    0,    # CCRet_0 [kmol/m3]
    300.0,  # TS_0 [K]
    370.0   # TjS_0 [K]
]

t_span = (0.0, 20.0)             # Time span in hours
t_eval = np.linspace(0, 20, 500) # Output evaluation points

# Solve using Radau (implicit Runge-Kutta method suitable for stiff DAE systems)
sol = solve_ivp(
    fun=dae_system,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    args=(params,),
    method='Radau',
    rtol=1e-6,
    atol=1e-8
)

# =============================================================================
# 4. PLOTTING RESULTS
# =============================================================================
if sol.success:
    print("Integration successful!")
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Volumes
    axs[0, 0].plot(sol.t, sol.y[0], label='$V_R$ (Reactor)')
    axs[0, 0].plot(sol.t, sol.y[6], label='$V_S$ (Separator)')
    axs[0, 0].set_title('Volumes')
    axs[0, 0].set_ylabel('Volume [$m^3$]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Reactor Concentrations
    axs[0, 1].plot(sol.t, sol.y[1], label='$C_{A1}$')
    axs[0, 1].plot(sol.t, sol.y[2], label='$C_{B1}$')
    axs[0, 1].plot(sol.t, sol.y[3], label='$C_{C1}$')
    axs[0, 1].set_title('Reactor Concentrations')
    axs[0, 1].set_ylabel('Concentration [$kmol/m^3$]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Retentate Concentrations
    axs[1, 0].plot(sol.t, sol.y[7], label='$C_{A,Ret}$')
    axs[1, 0].plot(sol.t, sol.y[8], label='$C_{B,Ret}$')
    axs[1, 0].plot(sol.t, sol.y[9], label='$C_{C,Ret}$')
    axs[1, 0].set_title('Separator Retentate Concentrations')
    axs[1, 0].set_xlabel('Time [h]')
    axs[1, 0].set_ylabel('Concentration [$kmol/m^3$]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Temperatures
    axs[1, 1].plot(sol.t, sol.y[4], label='$T_1$ (Reactor)')
    axs[1, 1].plot(sol.t, sol.y[5], label='$T_{j,R}$ (Jacket R)')
    axs[1, 1].plot(sol.t, sol.y[10], label='$T_S$ (Separator)')
    axs[1, 1].plot(sol.t, sol.y[11], label='$T_{j,S}$ (Jacket S)')
    axs[1, 1].set_title('Temperatures')
    axs[1, 1].set_xlabel('Time [h]')
    axs[1, 1].set_ylabel('Temperature [K]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print("Integration failed:", sol.message)