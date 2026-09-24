import numpy as np
import pandas as pd
import scipy
from scipy.sparse import lil_matrix, csr_matrix
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn



def variable_volume_cstr_with_jacket(t, state, params):
    V, CA, CB, CC, T, Tj = state

    # Inlet & Valve-driven Outlet Flows
    Fin = params[0]                     # Inlet volumetric flow (L/min)
    kc = params[1]                      # Valve linear flow coefficient (L^0.5 / min)
    V_sp = params[20]                   # Setpoint reactor volume (L)
    Fout = kc * (V-V_sp)+Fin            # Outlet volumetric flow (L/min)

    # Inlet Concentrations & Temperature
    CA0, CB0, CC0 = params[2], params[3], params[4]  # mol/L
    T0 = params[5]                     # K

    # Kinetic & Thermal Parameters
    A1, Ea1 = params[6], params[7]         # 1/min, J/mol
    A2, Ea2 = params[8], params[9]         # 1/min, J/mol
    R = 8.314                      # J/(mol*K)
    dH1, dH2 = params[10], params[11]  # J/mol
    rho, Cp = params[12], params[13]        # g/L, J/(g*K)

    # Dynamic Overall Heat Transfer Coefficient (scaled with liquid level V)
    U_a = params[14]                     # J/(min*K*L)
    UA = U_a * V                   # Heat capacity scaled with fill height

    # Jacket Parameters
    Vj = params[15]                      # Jacket volume (L)
    Fj = params[16]                      # Coolant flow (L/min)
    Tj0 = params[17]                    # Coolant inlet temp (K)
    rho_j, Cp_j = params[18], params[19]    # Coolant properties

    # Temperature-dependent Rate Constants
    k1 = A1 * np.exp(-Ea1 / (R * T))
    k2 = A2 * np.exp(-Ea2 / (R * T))
    r1, r2 = k1 * CA, k2 * CB

    # Differential Equations
    dV_dt = Fin - Fout
    dCA_dt = (Fin / V) * (CA0 - CA) - r1
    dCB_dt = (Fin / V) * (CB0 - CB) + r1 - r2
    dCC_dt = (Fin / V) * (CC0 - CC) + r2
    dT_dt = (Fin / V) * (T0 - T) + ((-dH1)*r1 + (-dH2)*r2)/(rho * Cp) - (UA / (rho * Cp * V)) * (T - Tj)
    dTj_dt = (Fj / Vj) * (Tj0 - Tj) + (UA / (rho_j * Cp_j * Vj)) * (T - Tj)

    return [dV_dt, dCA_dt, dCB_dt, dCC_dt, dT_dt, dTj_dt]

def variable_volume_simple_isothermal_cstr(t, state, params):
    V, CA, CB = state

    # Inlet & Valve-driven Outlet Flows
    Fin = params[0]                     # Inlet volumetric flow (L/min)
    kc = params[1]                      # Valve linear flow coefficient (L^0.5 / min)
    V_sp = params[2]                   # Setpoint reactor volume (L)
    Fout = kc * (V-V_sp)+Fin            # Outlet volumetric flow (L/min)

    # Inlet Concentrations & Temperature
    CA0, CB0 = params[3], params[4]  # mol/L

    # Kinetic Parameters and Rate Law
    k1 = params[5]
    k2 = params[6]
    r1 = k1 * CA
    

    # Differential Equations
    dV_dt = Fin - Fout
    dCA_dt = (Fin / V) * (CA0 - CA) - r1
    dCB_dt = (Fin / V) * (CB0 - CB) + r1

    return [dV_dt, dCA_dt, dCB_dt]


def build_bipartite_graph(equations, variables):
    """
    Build a bipartite graph linking equations to the variables they contain,
    returning both a networkx representation and a scipy sparse incidence matrix.

    Parameters
    ----------
    equations : list
        A list of sympy expressions/equations (e.g. sp.Eq(x + y, 1)),
        or strings that can be parsed by sympy.
    variables : list or set
        A collection of sympy symbols (or strings) representing the variables.

    Returns
    -------
    G : networkx.Graph
        Bipartite graph with equation nodes ('eq_0', 'eq_1', ...) and
        variable nodes (named by symbol). Equation nodes carry the
        original equation object as node data.
    M : scipy.sparse.csr_matrix
        Incidence matrix of shape (n_equations, n_variables), where
        M[i, j] = 1 if equation i contains variable j.
    var_order : list
        Ordered list of variable names corresponding to columns of M.
        (Row i of M corresponds to equations[i].)
    """
    # Preserve a stable order for variables (needed for matrix columns)
    var_order = [str(sp.Symbol(v) if isinstance(v, str) else v) for v in variables]
    var_set = {sp.Symbol(v) if isinstance(v, str) else v for v in variables}
    var_index = {name: idx for idx, name in enumerate(var_order)}

    n_eq = len(equations)
    n_var = len(var_order)

    G = nx.Graph()
    for var in var_order:
        G.add_node(var, bipartite="variable")

    M = lil_matrix((n_eq, n_var), dtype=np.int8)

    for i, eq in enumerate(equations):
        if isinstance(eq, str):
            eq = sp.sympify(eq)

        eq_node = f"eq_{i}"
        G.add_node(eq_node, bipartite="equation", equation=eq)

        eq_vars = eq.free_symbols & var_set
        for var in eq_vars:
            var_name = str(var)
            G.add_edge(eq_node, var_name)
            M[i, var_index[var_name]] = 1

    return G, M.tocsr(), var_order