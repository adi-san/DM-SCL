import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

np.random.seed(0)

params = {
    'F0': 10.0, 'CA0': 1.0, 'T0': 360.0, 'Fpg': 0.2, 'Fperm': 5.5,
    'rho': 1000.0, 'Cp': 4.184, 'rhoc': 1000.0, 'Cpc': 4.184, 'rhoH': 1000.0, 'CpH': 4.184,
    'dH1': -50000.0, 'dH2': -30000.0, 'A1': 1e6, 'EA1': 40000.0, 'A2': 5e5, 'EA2': 35000.0, 'R': 8.314,
    'dGA': 1000.0, 'dGB': 2000.0, 'dGC': 3000.0,
    'VjR': 2.0, 'FjR': 20.0, 'TCin': 290.0, 'VjS': 1.5, 'FH': 10.0, 'THin': 400.0,
    'UaR': 5000.0, 'UaS': 4000.0, 'kvR': 2.5, 'kvS': 2.0,
}
 
# ---------------- load data ----------------
data = pd.read_csv('training_data_heat_transfer_coefficients.csv')
print("Training data loaded successfully. Data shape:", data.shape)

# ---------------- constitutive terms (computed from CLEAN data) ----------------
clean = data.copy()
r1_clean = clean['CA1'] * params['A1'] * np.exp(-params['EA1'] / (params['R'] * clean['T1']))
r2_clean = clean['CB1'] * params['A2'] * np.exp(-params['EA2'] / (params['R'] * clean['T1']))
K_A_clean = np.exp(-params['dGA'] / (params['R'] * clean['TS']))
K_B_clean = np.exp(-params['dGB'] / (params['R'] * clean['TS']))
K_C_clean = np.exp(-params['dGC'] / (params['R'] * clean['TS']))
theta_QR_clean = params['UaR'] * clean['VR'] * (clean['T1'] - clean['TjR'])
theta_QS_clean = params['UaS'] * clean['VS'] * (clean['TjS'] - clean['TS'])
K_df_clean = pd.DataFrame({'K_A': K_A_clean, 'K_B': K_B_clean, 'K_C': K_C_clean})

# ---------------- add gaussian (multiplicative) noise to the OUTPUTS ----------------
# inputs stay clean; y = f(x) + f(x) * noise_level * N(0, 1)  (same as traditional_NNs.py)
NOISE_LEVEL = 0.03
def add_noise(y_clean):
    return y_clean + y_clean * NOISE_LEVEL * np.random.randn(*y_clean.shape)

r1, r2 = add_noise(r1_clean), add_noise(r2_clean)
theta_QR, theta_QS = add_noise(theta_QR_clean), add_noise(theta_QS_clean)
K_df = add_noise(K_df_clean)

# ---------------- GP helpers ----------------
def make_kernel(d):
    # ARD Matern-5/2: one lengthscale per input -> tells you which inputs matter
    return (C(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-2, (1e-8, 1.0)))
 
 
def fit_gp(X_df, y, y_clean, name, mode='plain', max_train=800, n_restarts=2, seed=0):
    """Clean inputs X_df -> noisy target(s) y; the WhiteKernel learns the output noise.
    Test R^2/RMSE are computed against y_clean; coverage is against the noisy y."""
    rng = np.random.RandomState(seed)
    X = X_df.values.astype(float)
    y_df = y.to_frame() if isinstance(y, pd.Series) else y
    out_names = [name] if isinstance(y, pd.Series) else list(y_df.columns)
    Y = y_df.values.astype(float)
    Yc = (y_clean.to_frame() if isinstance(y_clean, pd.Series) else y_clean).values.astype(float)
    n_out = Y.shape[1]
 
    # same 70/15/15 split as the NN script; GP trains on train+val (no early stopping needed)
    idx = rng.permutation(len(X))
    n_tr, n_va = int(0.7 * len(X)), int(0.15 * len(X))
    fit_idx, te = idx[:n_tr + n_va], idx[n_tr + n_va:]
    if len(fit_idx) > max_train:                       # exact GP is O(n^3)
        fit_idx = rng.choice(fit_idx, max_train, replace=False)
 
    xm, xs = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-12
    ym, ys = Y[fit_idx].mean(0), Y[fit_idx].std(0) + 1e-12
    Xtr, Ytr = (X[fit_idx] - xm) / xs, (Y[fit_idx] - ym) / ys
    Xte = (X[te] - xm) / xs
    d = X.shape[1]
 
    gp = GPR(make_kernel(d), alpha=1e-8, n_restarts_optimizer=n_restarts, random_state=seed).fit(Xtr, Ytr)

    # predict; GP std already includes the learned white-noise term
    mu, std = gp.predict(Xte, return_std=True)
    mu, std = mu.reshape(len(te), -1), np.asarray(std).reshape(len(te), -1)
    if std.shape[1] == 1 and n_out > 1:
        std = np.repeat(std, n_out, axis=1)
    pred, sd, true, noisy = mu * ys + ym, std * ys, Yc[te], Y[te]

    print(f"[{mode}:{name}] kernel: {gp.kernel_}")
    metrics = []
    for j, o in enumerate(out_names):
        err = pred[:, j] - true[:, j]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2 = float(1 - np.sum(err ** 2) / np.sum((true[:, j] - true[:, j].mean()) ** 2))
        cov = float(np.mean(np.abs(pred[:, j] - noisy[:, j]) <= 1.96 * sd[:, j]))
        print(f"[{mode}:{name}] TEST {o}: RMSE(clean) = {rmse:.5g}  R^2(clean) = {r2:.4f}  95% coverage(noisy) = {cov:.2f}")
        metrics.append({'target': o, 'model': mode, 'RMSE': rmse, 'R2': r2,
                        'cov95': cov, 'n_train': len(fit_idx), 'n_test': len(te)})
 
    fig, axes = plt.subplots(1, n_out, figsize=(4.8 * n_out, 4.2), squeeze=False)
    sel = rng.choice(len(te), min(200, len(te)), replace=False)
    for j, o in enumerate(out_names):
        a = axes[0, j]
        a.errorbar(true[sel, j], pred[sel, j], yerr=2 * sd[sel, j], fmt='o', ms=3,
                   alpha=0.5, elinewidth=0.6)
        lim = [min(true[:, j].min(), pred[:, j].min()), max(true[:, j].max(), pred[:, j].max())]
        a.plot(lim, lim, 'r--')
        a.set_xlabel('true (clean)')
        a.set_ylabel('GP mean (±2σ)')
        a.set_title(f'{o} [{mode}]  R²={metrics[j]["R2"]:.3f}, cov95={metrics[j]["cov95"]:.2f}')
    plt.tight_layout()
    plt.savefig(f'gp_{mode}_{name}_results.png', dpi=150)
    plt.close()
    return gp, (xm, xs, ym, ys), metrics
 
 
# ---------------- targets: name -> (clean input columns, noisy target(s), clean target(s)) ----------------
targets = {
    'theta_QR': (['VR', 'T1', 'TjR'], theta_QR, theta_QR_clean),
    'theta_QS': (['VS', 'TS', 'TjS'], theta_QS, theta_QS_clean),
    'r1':       (['CA1', 'T1'],       r1,       r1_clean),
    'r2':       (['CB1', 'T1'],       r2,       r2_clean),
    'K_ABC':    (['TS'],              K_df,     K_df_clean),   # 1 input -> 3 outputs (shared kernel)
}

results = []
mode = 'plain'      # NIGP dropped: inputs are clean, so there is no input noise to model
for name, (cols, y, y_clean) in targets.items():
    print(f"\n=== {mode.upper()} GP: {name}({', '.join(cols)}) ===")
    gp, sc, metrics = fit_gp(clean[cols], y, y_clean, name, mode=mode)
    results.extend(metrics)
    joblib.dump({'gp': gp, 'inputs': cols, 'xm': sc[0], 'xs': sc[1], 'ym': sc[2], 'ys': sc[3],
                 'outputs': list(y.columns) if isinstance(y, pd.DataFrame) else [name]},
                f'gp_{mode}_{name}.joblib')

summary = pd.DataFrame(results).set_index(['target', 'model'])
print("\n=== Test-set summary ===")
print(summary)
summary.to_csv('gp_summary.csv')
 
