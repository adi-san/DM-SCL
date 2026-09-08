from helper_functions import *

# Initial conditions: [V (L), CA, CB (mol/L)]
y0 = [1, 0.0, 0.0]
# Inlet & Valve-driven Outlet Flows
Fin = 12.0                     # Inlet volumetric flow (L/min)
kc = .25                       # Valve linear flow coefficient
# Inlet Concentrations & Temperature
CA0, CB0 = 2.0, 0.0  # mol/L
# Kinetic Parameters
k1, k2 = 0.1, 0.05  # 1/min
# reactor volume setpoint
V_sp = 12.0

# Pack parameters into a single array
params = [Fin, kc, V_sp, CA0, CB0, k1, k2]
# Time span for the simulation
t_span = (0, 120)
t_eval = np.linspace(0, 120, 600)
# Solve the ODEs
solution = solve_ivp(variable_volume_simple_isothermal_cstr, t_span, y0, args=([params]), t_eval=t_eval, method='RK45')
# Extract the results
V, CA, CB = solution.y
# Plotting Dynamic Profiles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Volume Profile
ax1.plot(solution.t, V, label='Volume (L)', color='blue')
ax1.set_ylabel('Volume $V$ (L)')
ax1.set_title('Transient Response of Variable-Volume Simple Isothermal CSTR')
ax1.grid(True, linestyle='--')

# Concentrations Profile
ax2.plot(solution.t, CA, label='CA (mol/L)', color='red')
ax2.plot(solution.t, CB, label='CB (mol/L)', color='green')
ax2.set_ylabel('Concentration (mol/L)')
ax2.set_xlabel('Time (min)')
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()