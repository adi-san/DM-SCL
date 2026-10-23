"""
Description:
I want to compare 2 approaches
    # GP trained assuming that I have liberty to measure everything , uses data from noisy_states_algebraics_for_all.csv
    # GP trained using DM Decomp assisted structural exhaustion, uses data from noisy_heat_r_kappa_CCPerm_DM.csv
Rough Algorithm Workflow
1) read the 2 csvs as aforementioned
2) There are 7 functions I want to learn with the GPs. Their inputs are within the parentheses
    1) theta_QR(VR,T1,TjR)
    2) theta_QS(VS, TS, TjS)
    3) r1(CA1,T1)
    4) r2(CB1,T1)
    5) K_A(TS)
    6) K_B(TS)
    7) K_C(TS)
    - it is probably best to use 5 separate GPs, one each for theta_QR, theta_QS, r1, r2, and one for all the K terms as they depend on the same input
    - Train the GPs using data from their respective CSV files. This means a total of 10 GPs trained
        - might have to use k-fold cross validation if results are subpar with a traditional GP approach
3) for each trained GP, evaluate on a small grid of states that span the values used for training. Compare to the true noise free results in a
parity plot with uncertainty bars included
4) using the gp surrogates associated with the noisy_states_algebraics_for_all and noisy_heat_r_kappa_CCPerm_DM data respectively:
    - test the dynamic response of the models against the true simulation using various step inputs in F0, FjR, and FH from a steady state
    - The initial conditions of the states will be their values at that steady state
    - plot the results, showing uncertainty in the GP
"""
import itertools
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.model_selection import KFold

np.random.seed(0)
warnings.filterwarnings('ignore', category=ConvergenceWarning)   # ls hitting its upper bound just means "input is irrelevant"
warnings.filterwarnings('ignore', category=RuntimeWarning)       # all-NaN slices when a surrogate draw fails to integrate

params = {
    'F0': 10.0, 'CA0': 1.0, 'T0': 360.0, 'Fpg': 0.2, 'Fperm': 5.5,
    'rho': 1000.0, 'Cp': 4.184, 'rhoc': 1000.0, 'Cpc': 4.184, 'rhoH': 1000.0, 'CpH': 4.184,
    'dH1': -50000.0, 'dH2': -30000.0, 'A1': 1e6, 'EA1': 40000.0, 'A2': 5e5, 'EA2': 35000.0, 'R': 8.314,
    'dGA': 1000.0, 'dGB': 2000.0, 'dGC': 3000.0,
    'VjR': 2.0, 'FjR': 20.0, 'TCin': 290.0, 'VjS': 1.5, 'FH': 10.0, 'THin': 400.0,
    'UaR': 5000.0, 'UaS': 4000.0, 'kvR': 2.5, 'kvS': 2.0,
}
STATES = ['VR', 'CA1', 'CB1', 'CC1', 'T1', 'TjR', 'VS', 'CARet', 'CBRet', 'CCRet', 'TS', 'TjS']
UNITS = ['m³', 'kmol/m³', 'kmol/m³', 'kmol/m³', 'K', 'K', 'm³', 'kmol/m³', 'kmol/m³', 'kmol/m³', 'K', 'K']

# plot style: approach colors stay with the approach in every figure; the true model is always neutral ink
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COLORS = {'all': '#2a78d6', 'DM': '#eb6834'}
LABELS = {'all': 'all measured', 'DM': 'DM decomp'}
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.frameon': False, 'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
})

# ---------------- 1) read the data ----------------
# NOTE: paths are relative to the working directory (like the other scripts), so run this from the folder that holds
# the CURRENT csvs (the workspace root, "DM SCL"); the copies inside DM-SCL/ are stale (different FjR levels).
APPROACHES = {'all': 'noisy_states_algebraics_for_all.csv', 'DM': 'noisy_heat_r_kappa_CCPerm_DM.csv'}
data = {a: pd.read_csv(f) for a, f in APPROACHES.items()}
clean = pd.read_csv('simulator_data_no_noise.csv')        # noise-free simulation: used only for ranges / a steady-state guess
for a, f in APPROACHES.items():
    print(f"[{a}] {f}: {data[a].shape[0]} training rows")
state_range = (clean[STATES].max() - clean[STATES].min()).values

# GP name -> (input columns, output columns). K_A, K_B, K_C share the input TS, so they share one GP
tasks = {
    'theta_QR': (['VR', 'T1', 'TjR'], ['theta_QR']),
    'theta_QS': (['VS', 'TS', 'TjS'], ['theta_QS']),
    'r1':       (['CA1', 'T1'],       ['r1']),
    'r2':       (['CB1', 'T1'],       ['r2']),
    'K_ABC':    (['TS'],              ['K_A', 'K_B', 'K_C']),
}
OUT_NAMES = [o for _, outs in tasks.values() for o in outs]

# the true (noise-free) constitutive laws; X columns follow the input order of each GP above
true_fns = {
    'theta_QR': lambda X: (params['UaR'] * X[:, 0] * (X[:, 1] - X[:, 2]))[:, None],
    'theta_QS': lambda X: (params['UaS'] * X[:, 0] * (X[:, 2] - X[:, 1]))[:, None],
    'r1':       lambda X: (params['A1'] * np.exp(-params['EA1'] / (params['R'] * X[:, 1])) * X[:, 0])[:, None],
    'r2':       lambda X: (params['A2'] * np.exp(-params['EA2'] / (params['R'] * X[:, 1])) * X[:, 0])[:, None],
    'K_ABC':    lambda X: np.exp(-np.array([params['dGA'], params['dGB'], params['dGC']]) / (params['R'] * X[:, [0]])),
}


# ---------------- 2) GP machinery ----------------
def make_kernel(d, noise_floor):
    # ARD Matern-5/2 + WhiteKernel to learn the output noise (same kernel as traditional_GPs.py), with a floor on the noise
    return (C(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(max(1e-2, noise_floor), (noise_floor, 1.0)))


class GPSurrogate:
    """A fitted sklearn GP (standardized inputs/outputs) with a fast numpy posterior mean, the std of the noise-free
    function, and pathwise posterior function draws. sklearn's predict() is too slow to call inside an ODE right-hand
    side, and the draws give *consistent* function samples along a whole trajectory (needed to propagate GP uncertainty
    through the dynamics)."""
    def __init__(self, gp, xm, xs, ym, ys):
        self.gp, self.xm, self.xs, self.ym, self.ys = gp, xm, xs, ym, ys
        k = gp.kernel_
        self.sf2, self.ls, self.noise = k.k1.k1.constant_value, np.atleast_1d(k.k1.k2.length_scale), k.k2.noise_level
        self.Xtr, self.n_out = gp.X_train_, len(ym)
        self.Ytr = gp.y_train_.reshape(len(self.Xtr), -1)
        self.alpha = gp.alpha_.reshape(len(self.Xtr), -1)
        self.Kn = k(self.Xtr) + gp.alpha * np.eye(len(self.Xtr))        # K + noise*I (standardized units)

    def _k(self, A, B):                                                 # Matern-5/2 ARD kernel, without the noise term
        r = np.sqrt((((A[:, None, :] - B[None, :, :]) / self.ls) ** 2).sum(-1))
        return self.sf2 * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)

    def mean(self, X):                                                  # raw X (n, d) -> (n, n_out)
        Z = (np.atleast_2d(X) - self.xm) / self.xs
        return self.ym + self.ys * (self._k(Z, self.Xtr) @ self.alpha)

    def latent_std(self, X):                                            # std of the NOISE-FREE function f(x), (n, n_out)
        Z = (np.atleast_2d(X) - self.xm) / self.xs
        Ks = self._k(Z, self.Xtr)
        var = self.sf2 - np.einsum('ij,ji->i', Ks, np.linalg.solve(self.Kn, Ks.T))
        return np.sqrt(np.maximum(var, 0))[:, None] * self.ys

    def draw(self, rng, M=2000):
        """One joint posterior sample of the function (pathwise conditioning, Wilson et al. 2020): a random-Fourier-feature
        prior sample corrected by the training data. Returns f(X) with the same call signature as mean()."""
        d, n = len(self.ls), len(self.Xtr)
        # Matern-5/2 spectral density = Student-t with 5 dof, scaled by 1/lengthscale
        W = rng.standard_normal((self.n_out, M, d)) * np.sqrt(5 / rng.chisquare(5, (self.n_out, M, 1))) / self.ls
        b = rng.uniform(0, 2 * np.pi, (self.n_out, M))
        th = rng.standard_normal((self.n_out, M))
        amp = np.sqrt(2 * self.sf2 / M)
        phi = lambda Z, o: amp * np.cos(Z @ W[o].T + b[o])
        V = np.empty((n, self.n_out))
        for o in range(self.n_out):
            eps = np.sqrt(self.noise) * rng.standard_normal(n)
            V[:, o] = np.linalg.solve(self.Kn, self.Ytr[:, o] - phi(self.Xtr, o) @ th[o] - eps)

        def f(X):
            Z = (np.atleast_2d(X) - self.xm) / self.xs
            prior = np.stack([phi(Z, o) @ th[o] for o in range(self.n_out)], -1)
            return self.ym + self.ys * (prior + self._k(Z, self.Xtr) @ V)
        return f


def fit_gp(X, Y, noise_floor=1e-8, n_restarts=10, seed=0):
    xm, xs, ym, ys = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    gp = GPR(make_kernel(X.shape[1], noise_floor), alpha=1e-8, n_restarts_optimizer=n_restarts, random_state=seed)
    gp.fit((X - xm) / xs, (Y - ym) / ys)
    return GPSurrogate(gp, xm, xs, ym, ys)


# A plain marginal-likelihood fit is fragile on these 45 rows: in the DM data theta_QS and TjS are algebraically tied
# through the jacket balance (theta_QS = rho*Cp*FH*(THin - TjS)), so the likelihood can prefer a noise-free, tiny-lengthscale
# interpolant that does not generalize (and which optimum is found depends on the number of restarts). So the noise floor of
# the WhiteKernel is chosen by k-fold CV: the smallest floor whose CV error is within 2% of the best, i.e. the plain GP
# (floor 1e-8) is kept unless CV says it is subpar.
NOISE_FLOORS = [1e-8, 1e-3, 1e-2, 3e-2, 1e-1]


def cv_rmse(X, Y, noise_floor, k=5, seed=0):
    err = []
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m = fit_gp(X[tr], Y[tr], noise_floor, n_restarts=2)
        err.append((m.mean(X[te]) - Y[te]) / Y.std(0))                  # held-out error in units of the target std
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def select_noise_floor(X, Y, tol=0.02):
    scores = [cv_rmse(X, Y, f) for f in NOISE_FLOORS]
    best = min(scores)
    return next(f for f, s in zip(NOISE_FLOORS, scores) if s <= best * (1 + tol)), scores


# ---------------- 3) train the 10 GPs (5 per approach) ----------------
models, train_rows = {}, []
for approach, df in data.items():
    models[approach] = {}
    for name, (cols, outs) in tasks.items():
        X, Y = df[cols].values.astype(float), df[outs].values.astype(float)
        floor, scores = select_noise_floor(X, Y)
        m = fit_gp(X, Y, floor, n_restarts=10)
        models[approach][name] = m
        print(f"[{approach}:{name}({', '.join(cols)})] noise floor {floor:g} (5-fold CV RMSE/std: "
              + ", ".join(f"{s:.3f}" for s in scores) + f")\n    kernel: {m.gp.kernel_}")
        train_rows.append({'approach': approach, 'gp': name, 'n_train': len(X), 'noise_floor': floor,
                           **{f'cv_rmse_floor_{f:g}': s for f, s in zip(NOISE_FLOORS, scores)},
                           'kernel': str(m.gp.kernel_)})
        joblib.dump({'gp': m.gp, 'inputs': cols, 'outputs': outs, 'xm': m.xm, 'xs': m.xs, 'ym': m.ym, 'ys': m.ys,
                     'noise_floor': floor}, f'gp_{approach}_{name}.joblib')
pd.DataFrame(train_rows).to_csv('gp_train_summary.csv', index=False)


# ---------------- 4) the true model, with pluggable constitutive terms ----------------
def make_cons(fns):
    """fns: GP name -> callable(X (1, d)) -> (1, n_out). Returns cons(y) -> (theta_QR, theta_QS, r1, r2, K_A, K_B, K_C)."""
    idx = {n: [STATES.index(c) for c in cols] for n, (cols, _) in tasks.items()}
    return lambda y: np.concatenate([fns[n](y[idx[n]][None, :])[0] for n in tasks])


true_cons = make_cons(true_fns)


def rhs(t, y, p, cons):
    # same balances as reactor_separator_dae.py; only the constitutive terms come from cons()
    VR, CA1, CB1, CC1, T1, TjR, VS, CARet, CBRet, CCRet, TS, TjS = y
    F1 = p['kvR'] * np.sqrt(max(VR, 1e-6))
    FRet = p['kvS'] * np.sqrt(max(VS, 1e-6))
    FR = p['Fperm'] - p['Fpg']
    theta_QR, theta_QS, r1, r2, K_A, K_B, K_C = cons(y)
    CAPerm, CBPerm, CCPerm = K_A * CARet, K_B * CBRet, K_C * CCRet

    dVR_dt = p['F0'] + FR - F1
    dCA1_dt = (p['F0'] / VR) * (p['CA0'] - CA1) + (FR / VR) * (CAPerm - CA1) - r1
    dCB1_dt = -(p['F0'] / VR) * CB1 + (FR / VR) * (CBPerm - CB1) + r1 - r2
    dCC1_dt = -(p['F0'] / VR) * CC1 + (FR / VR) * (CCPerm - CC1) + r2
    dT1_dt = ((p['F0'] * (p['T0'] - T1) + FR * (TS - T1)) / VR
              + (-p['dH1'] * r1 - p['dH2'] * r2) / (p['rho'] * p['Cp'])
              - theta_QR / (p['rho'] * p['Cp'] * VR))
    dTjR_dt = (p['FjR'] / p['VjR']) * (p['TCin'] - TjR) + theta_QR / (p['rhoc'] * p['Cpc'] * p['VjR'])
    dVS_dt = F1 - FRet - FR - p['Fpg']
    dCARet_dt = (F1 / VS) * (CA1 - CARet) + ((FR + p['Fpg']) / VS) * (CARet - CAPerm)
    dCBRet_dt = (F1 / VS) * (CB1 - CBRet) + ((FR + p['Fpg']) / VS) * (CBRet - CBPerm)
    dCCRet_dt = (F1 / VS) * (CC1 - CCRet) + ((FR + p['Fpg']) / VS) * (CCRet - CCPerm)
    dTS_dt = (F1 / VS) * (T1 - TS) + theta_QS / (p['rho'] * p['Cp'] * VS)
    dTjS_dt = (p['FH'] / p['VjS']) * (p['THin'] - TjS) - theta_QS / (p['rhoH'] * p['CpH'] * p['VjS'])
    return [dVR_dt, dCA1_dt, dCB1_dt, dCC1_dt, dT1_dt, dTjR_dt,
            dVS_dt, dCARet_dt, dCBRet_dt, dCCRet_dt, dTS_dt, dTjS_dt]


def steady_state(F0, FjR, FH, guess):
    p = {**params, 'F0': F0, 'FjR': FjR, 'FH': FH}
    sol, info, ier, msg = fsolve(lambda y: rhs(0, y, p, true_cons), guess, xtol=1e-12, full_output=True)
    return (sol if ier == 1 else None), p


# ---------------- 5) parity tests on grids of states ----------------
# (a) Steady-state grid: true steady states at a grid of (F0, FjR, FH) that lies inside the training ranges but not on the
#     training levels (F0 2-10, FjR 8-80, FH 10-40), so every grid state is physically reachable and held out. Rows are
#     dropped by the same temperature-approach filter (>= 10 K) used to build the training data.
# (b) Box grid: every combination of the min..max range of each GP input (what GP_with _noisy_heat_data.py did). This
#     includes impossible states (e.g. T1 < TjR) the GPs have never seen, so it mostly measures extrapolation.
def steady_state_grid():
    rows, guess = [], clean[STATES].values[len(clean) // 2]
    for F0, FjR, FH in itertools.product(np.linspace(2.5, 9.5, 5), np.geomspace(10, 70, 5), np.linspace(14, 38, 4)):
        y, _ = steady_state(F0, FjR, FH, guess)
        if y is not None and abs(y[4] - y[5]) >= 10 and abs(y[10] - y[11]) >= 10:
            rows.append(dict(zip(STATES, y)))
            guess = y
    return pd.DataFrame(rows)


def box_grid(cols, n_per_dim={1: 30, 2: 15, 3: 6}):
    axes_1d = [np.linspace(clean[c].min(), clean[c].max(), n_per_dim[len(cols)]) for c in cols]
    return np.stack([m.ravel() for m in np.meshgrid(*axes_1d, indexing='ij')], axis=1)


G = steady_state_grid()
print(f"\nsteady-state test grid: {len(G)} states")
grids = {'ss': ('steady-state grid', {n: G[cols].values for n, (cols, _) in tasks.items()}),
         'box': ('box grid', {n: box_grid(cols) for n, (cols, _) in tasks.items()})}

parity_rows = []
for gkey, (gname, X_grid) in grids.items():
    for approach in data:
        res = {}                                                      # output -> (true, mean, latent std)
        for name, (cols, outs) in tasks.items():
            X = X_grid[name]
            mu, sd, yt = models[approach][name].mean(X), models[approach][name].latent_std(X), true_fns[name](X)
            for j, o in enumerate(outs):
                res[o] = (yt[:, j], mu[:, j], sd[:, j])
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
        for ax, o in zip(axes.ravel(), OUT_NAMES):
            yt, mu, sd = res[o]
            err = mu - yt
            r2 = 1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            cov = float(np.mean(np.abs(err) <= 2 * sd))
            parity_rows.append({'grid': gkey, 'approach': approach, 'output': o, 'n_points': len(yt), 'R2': r2,
                                'RMSE': rmse, 'RMSE_over_std': rmse / yt.std(), 'cov95': cov})
            ax.errorbar(yt, mu, yerr=2 * sd, fmt='o', ms=4, color=COLORS[approach], mec=SURFACE, mew=0.6,
                        ecolor=COLORS[approach], elinewidth=0.8, alpha=0.8, zorder=3)
            lim = [min(yt.min(), mu.min()), max(yt.max(), mu.max())]
            pad = 0.03 * (lim[1] - lim[0])
            ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color=INK2, lw=1, zorder=2)
            ax.set_xlabel('true (noise-free)')
            ax.set_ylabel('GP mean ± 2σ')
            ax.set_title(f'{o}   R² = {r2:.3f},  inside 2σ: {cov:.0%}')
            if o.startswith('K_'):
                ax.ticklabel_format(useOffset=False, style='plain')
        info = axes.ravel()[-1]
        info.axis('off')
        info.text(0, 1, f"GP trained on:\n{APPROACHES[approach]}\n\ntest: {gname}\n({len(G) if gkey == 'ss' else 'min..max box'}"
                        f"{' states' if gkey == 'ss' else ''})\n\nmarkers: GP posterior mean\nbars: ±2σ of the noise-free\n"
                        f"function (not the measurement noise)\nline: parity", va='top', fontsize=9, color=INK2)
        fig.suptitle(f'GP vs true model, {LABELS[approach]} approach, {gname}', fontsize=12, x=0.01, ha='left')
        plt.tight_layout()
        fname = f"gp_parity_{approach}.png" if gkey == 'ss' else f"gp_parity_box_{approach}.png"
        plt.savefig(fname, dpi=130)
        plt.close(fig)
        print(f"saved {fname}")
parity = pd.DataFrame(parity_rows)
parity.to_csv('gp_parity_summary.csv', index=False)
print("\n=== parity summary on the steady-state grid (R^2 / fraction of truth inside 2σ) ===")
tab = parity[parity.grid == 'ss'].pivot(index='output', columns='approach', values=['R2', 'cov95']).loc[OUT_NAMES]
print(tab.round(3).to_string())
print("\n=== parity summary on the box grid (R^2) ===")
print(parity[parity.grid == 'box'].pivot(index='output', columns='approach', values='R2').loc[OUT_NAMES].round(3).to_string())

# ---------------- 6) dynamic response: step inputs from a steady state ----------------
nominal = (6.5, 30.0, 25.0)                                          # (F0, FjR, FH); off the training levels, inside the ranges
scenarios = {                                                        # key -> (title, (F0, FjR, FH) after the step at t = 0)
    'F0_up':      ('F0 step 6.5 → 9',                       (9.0, 30.0, 25.0)),
    'F0_down':    ('F0 step 6.5 → 4',                       (4.0, 30.0, 25.0)),
    'FjR_up':     ('FjR step 30 → 60',                      (6.5, 60.0, 25.0)),
    'FjR_down':   ('FjR step 30 → 12',                      (6.5, 12.0, 25.0)),
    'FH_up':      ('FH step 25 → 40',                       (6.5, 30.0, 40.0)),
    'FH_down':    ('FH step 25 → 12',                       (6.5, 30.0, 12.0)),
    'combined':   ('F0 6.5 → 8, FjR 30 → 15, FH 25 → 35',   (8.0, 15.0, 35.0)),
}
t_eval = np.linspace(0, 20, 201)                                     # hours
N_MC = 100                                                           # GP posterior draws per surrogate and scenario
y_ss, _ = steady_state(*nominal, clean[STATES].values[len(clean) // 2])
assert y_ss is not None
print(f"\nnominal steady state (F0, FjR, FH) = {nominal}: " + ", ".join(f"{s}={v:.4g}" for s, v in zip(STATES, y_ss)))


def simulate(p, y0, cons, rtol=1e-6, atol=1e-8):
    out = np.full((len(STATES), len(t_eval)), np.nan)                # stays NaN after a failed integration
    try:
        s = solve_ivp(rhs, (0, t_eval[-1]), y0, t_eval=t_eval, args=(p, cons), method='Radau', rtol=rtol, atol=atol)
        out[:, :s.y.shape[1]] = s.y
    except Exception:
        pass
    return out


dyn_rows = []
for si, (skey, (stitle, step)) in enumerate(scenarios.items()):
    p = {**params, 'F0': step[0], 'FjR': step[1], 'FH': step[2]}     # inputs jump at t = 0; states start at the nominal steady state
    truth = simulate(p, y_ss, true_cons, rtol=1e-8, atol=1e-10)
    traj = {}
    for approach in data:
        rng = np.random.default_rng(100 + si)
        mean_traj = simulate(p, y_ss, make_cons({n: m.mean for n, m in models[approach].items()}))
        draws = np.stack([simulate(p, y_ss, make_cons({n: m.draw(rng) for n, m in models[approach].items()}))
                          for _ in range(N_MC)])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        traj[approach] = (mean_traj, lo, hi)
        n_failed = int(np.sum(np.isnan(draws).any(axis=(1, 2))))
        nrmse = np.sqrt(np.nanmean((mean_traj - truth) ** 2, axis=1)) / state_range
        inside = np.nanmean((truth >= lo) & (truth <= hi), axis=1)
        for k, s in enumerate(STATES):
            dyn_rows.append({'scenario': skey, 'approach': approach, 'state': s, 'NRMSE_mean_surrogate': nrmse[k],
                             'frac_true_inside_95band': inside[k], 'failed_draws': n_failed})

    fig, axes = plt.subplots(4, 3, figsize=(13, 13), sharex=True)
    for k, (ax, s, u) in enumerate(zip(axes.ravel(), STATES, UNITS)):
        for approach in data:
            mean_traj, lo, hi = traj[approach]
            ax.fill_between(t_eval, lo[k], hi[k], color=COLORS[approach], alpha=0.25, lw=0, zorder=2)
            ax.plot(t_eval, mean_traj[k], color=COLORS[approach], lw=1.6, zorder=3)
        ax.plot(t_eval, truth[k], color=INK, lw=1.6, zorder=4)
        # a state that does not move (e.g. VR, VS under an FjR or FH step) would autoscale to numerical noise
        ymin = np.nanmin([np.nanmin(truth[k])] + [np.nanmin(traj[a][1][k]) for a in data])
        ymax = np.nanmax([np.nanmax(truth[k])] + [np.nanmax(traj[a][2][k]) for a in data])
        if ymax - ymin < 0.05 * state_range[k]:
            ax.set_ylim((ymax + ymin) / 2 - 0.025 * state_range[k], (ymax + ymin) / 2 + 0.025 * state_range[k])
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.set_title(f'{s}  [{u}]')
        if k >= 9:
            ax.set_xlabel('time [h]')
    handles = [Line2D([0], [0], color=INK, lw=1.8, label='true simulation')]
    handles += [Line2D([0], [0], color=COLORS[a], lw=1.8, label=f'GP surrogate, {LABELS[a]} (mean; band = 95% of {N_MC} posterior draws)')
                for a in data]
    fig.suptitle(f'Dynamic response from the steady state at (F0, FjR, FH) = {nominal}:  {stitle}', fontsize=12, x=0.01, ha='left')
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.965), ncol=1, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(f'gp_dynamic_{skey}.png', dpi=130)
    plt.close(fig)
    print(f"saved gp_dynamic_{skey}.png")

dyn = pd.DataFrame(dyn_rows)
dyn.to_csv('gp_dynamic_summary.csv', index=False)
gp_states = [s for s in STATES if s not in ('VR', 'VS')]               # VR, VS do not involve a GP (hydraulics only)
d = dyn[dyn.state.isin(gp_states)].groupby(['scenario', 'approach']).agg(
    NRMSE=('NRMSE_mean_surrogate', 'mean'), inside95=('frac_true_inside_95band', 'mean'), failed=('failed_draws', 'max'))
print(f"\n=== dynamic test, averaged over the {len(gp_states)} states that depend on a GP ===")
print("NRMSE = RMSE of the GP-mean surrogate vs true, divided by the state's range in the training data;")
print("inside95 = fraction of time the true trajectory lies inside the 95% band of the GP posterior draws")
print(d.unstack('approach').loc[list(scenarios)].round(3).to_string())
