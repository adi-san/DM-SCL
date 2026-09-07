from helper_functions import *

# Initial conditions: [V (L), CA, CB, CC (mol/L), T, Tj (K)]
y0 = [1, 0.0, 0.0, 0.0, 300.0, 285.0]
t_span = (0, 120)
t_eval = np.linspace(0, 120, 600)

# Inlet & Valve-driven Outlet Flows
Fin = 12.0                     # Inlet volumetric flow (L/min)
kc = .25                       # Valve linear flow coefficient 
# Inlet Concentrations & Temperature
CA0, CB0, CC0 = 2.0, 0.0, 0.0  # mol/L
T0 = 300.0

# Kinetic & Thermal Parameters
A1, Ea1 = 1e7, 45000.0         # 1/min, J/mol
A2, Ea2 = 5e7, 50000.0         # 1/min, J/mol
R = 8.314                      # J/(mol*K)
dH1, dH2 = -50000.0, -40000.0  # J/mol
rho, Cp = 1000.0, 4.184        # g/L, J/(g*K)

# Dynamic Overall Heat Transfer Coefficient
U_a = 80.0                     # J/(min*K*L)

# Jacket Parameters

Vj = 20.0                      # Jacket volume (L)
Fj = 6.0                      # Coolant flow (L/min)
Tj0 = 285.0                    # Coolant inlet temp (K)
rho_j, Cp_j = 1000.0, 4.184    # Coolant properties

# reactor volume setpoint
V_sp = 12.0

# Pack parameters into a single array
params = [Fin, kc, CA0, CB0, CC0, T0,A1, Ea1, A2, Ea2, dH1, dH2, rho, Cp,U_a, Vj, Fj, Tj0, rho_j, Cp_j,V_sp]
# Solve the ODEs
solution = solve_ivp(variable_volume_cstr_with_jacket, t_span, y0, args=([params]), t_eval=t_eval, method='RK45')

# Extract the results
V, CA, CB, CC, T, Tj = solution.y
# Plotting Dynamic Profiles
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Volume Profile
ax1.plot(solution.t, V, label='Volume (L)', color='blue')
ax1.set_ylabel('Volume $V$ (L)')
ax1.set_title('Transient Response of Variable-Volume CSTR')
ax1.grid(True, linestyle='--')

# concentrations Profile
ax2.plot(solution.t, CA, label='CA (mol/L)', color='red')
ax2.plot(solution.t, CB, label='CB (mol/L)', color='green')
ax2.plot(solution.t, CC, label='CC (mol/L)', color='orange')
ax2.set_ylabel('Concentration (mol/L)')
ax2.legend()
ax2.grid(True, linestyle='--')

# Temperature Profile
ax3.plot(solution.t, T, label='Reactor Temp $T$ (K)', color='purple')
ax3.plot(solution.t, Tj, label='Jacket Temp $T_j$ (K)', color='brown')
ax3.set_xlabel('Time (min)')
ax3.set_ylabel('Temperature (K)')
ax3.legend()
ax3.grid(True, linestyle='--')

plt.tight_layout()
plt.show()