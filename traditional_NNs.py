from scipy.optimize import fsolve
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


torch.manual_seed(0)
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
 

 
# ---------------- add gaussian noise ----------------
clean = data.copy()
noise_level = 0.03  # 0.5%

# for col in ['VR', 'VS', 'TjR', 'TjS', 'T1', 'TS', 'CA1', 'CB1']:
#     data[col] += data[col] * noise_level * np.random.randn(len(data))

# ---------------- constitutive terms (computed from CLEAN data) ----------------
r1_clean = clean['CA1'] * params['A1'] * np.exp(-params['EA1'] / (params['R'] * clean['T1']))
r2_clean = clean['CB1'] * params['A2'] * np.exp(-params['EA2'] / (params['R'] * clean['T1']))
K_A_clean = np.exp(-params['dGA'] / (params['R'] * clean['TS']))
K_B_clean = np.exp(-params['dGB'] / (params['R'] * clean['TS']))
K_C_clean = np.exp(-params['dGC'] / (params['R'] * clean['TS']))
theta_QR_clean = params['UaR'] * clean['VR'] * (clean['T1'] - clean['TjR'])
theta_QS_clean = params['UaS'] * clean['VS'] * (clean['TjS'] - clean['TS'])

# add noise to the outputs eg y=f(x) + noise
r1 = r1_clean + r1_clean * noise_level * np.random.randn(len(clean))
r2 = r2_clean + r2_clean * noise_level * np.random.randn(len(clean))
K_A = K_A_clean + K_A_clean * noise_level * np.random.randn(len(clean))
K_B = K_B_clean + K_B_clean * noise_level * np.random.randn(len(clean))
K_C = K_C_clean + K_C_clean * noise_level * np.random.randn(len(clean))
theta_QR = theta_QR_clean + theta_QR_clean * noise_level * np.random.randn(len(clean))
theta_QS = theta_QS_clean + theta_QS_clean * noise_level * np.random.randn(len(clean))

# # clean outputs (no noise) for training the NNs
# r1 = r1_clean
# r2 = r2_clean
# K_A = K_A_clean
# K_B = K_B_clean
# K_C = K_C_clean
# theta_QR = theta_QR_clean
# theta_QS = theta_QS_clean

# ---------------- plot: with vs without noise ----------------
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
noise_pairs = {'r1': (r1_clean, r1), 'r2': (r2_clean, r2), 'K_A': (K_A_clean, K_A), 'K_B': (K_B_clean, K_B),
               'K_C': (K_C_clean, K_C), 'theta_QR': (theta_QR_clean, theta_QR), 'theta_QS': (theta_QS_clean, theta_QS)}
for ax, (col, (yc, yn)) in zip(axes.ravel(), noise_pairs.items()):
    ax.plot(np.asarray(yc), label='clean', lw=1.5)
    ax.plot(np.asarray(yn), label='noisy', lw=0.7, alpha=0.7)
    ax.set_title(col)
axes[0, 0].legend()
plt.tight_layout()
plt.savefig('noise_comparison.png', dpi=150)
plt.close()
 
# ---------------- NN definition ----------------
class MLP(nn.Module):
    def __init__(self, n_in, n_out=1, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Sigmoid(),
            nn.Linear(hidden, hidden), nn.Sigmoid(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)
 
def train_theta(X_df, y, y_clean, name, epochs=3000, lr=1e-3, seed=0):
    """Train on clean inputs X_df -> noisy target(s) y (Series = 1 output, DataFrame = multiple outputs).
    Test metrics are computed against y_clean (same shape as y)."""
    rng = np.random.RandomState(seed)
    X = X_df.values.astype(np.float32)
    y_df = y.to_frame() if isinstance(y, pd.Series) else y
    yc_df = y_clean.to_frame() if isinstance(y_clean, pd.Series) else y_clean
    Y_clean = yc_df.values.astype(np.float32)
    out_names = [str(c) if c is not None else name for c in y_df.columns]
    if isinstance(y, pd.Series):
        out_names = [name]
    Y = y_df.values.astype(np.float32)          # (n, n_out)
    n_out = Y.shape[1]
 
    # 70/15/15 split
    idx = rng.permutation(len(X))
    n_tr, n_va = int(0.7 * len(X)), int(0.15 * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
 
    # standardize using training statistics only (each output scaled separately)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-12
    f = lambda a, m, s: torch.tensor((a - m) / s)
    Xtr, Xva, Xte = f(X[tr], xm, xs), f(X[va], xm, xs), f(X[te], xm, xs)
    ytr, yva = f(Y[tr], ym, ys), f(Y[va], ym, ys)
 
    model = MLP(X.shape[1], n_out)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100)
    loss_fn = nn.MSELoss()   # mean over all outputs (each already unit-variance)
 
    best, best_state, hist = np.inf, None, {'train': [], 'val': []}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), yva).item()
        sched.step(v)
        hist['train'].append(loss.item()); hist['val'].append(v)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 200 == 0:
            print(f"[{name}] epoch {ep:5d}  train {loss.item():.4f}  val {v:.4f}")
    model.load_state_dict(best_state)
 
    # test evaluation per output (physical units)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy() * ys + ym
    true = Y_clean[te]      # score against the underlying noise-free function
    noisy = Y[te]

    metrics = []
    for j, oname in enumerate(out_names):
        rmse = float(np.sqrt(np.mean((pred[:, j] - true[:, j]) ** 2)))
        r2 = float(1 - np.sum((pred[:, j] - true[:, j]) ** 2) / np.sum((true[:, j] - true[:, j].mean()) ** 2))
        rmse_noisy = float(np.sqrt(np.mean((pred[:, j] - noisy[:, j]) ** 2)))
        print(f"[{name}] TEST {oname}: RMSE(clean) = {rmse:.5g}  R^2(clean) = {r2:.4f}  RMSE(noisy) = {rmse_noisy:.5g}")
        metrics.append({'target': oname, 'network': name, 'RMSE': rmse, 'R2': r2,
                        'RMSE_noisy': rmse_noisy, 'n_test': len(te)})
 
    fig, ax = plt.subplots(1, 1 + n_out, figsize=(4.5 * (1 + n_out), 4))
    ax[0].semilogy(hist['train'], label='train'); ax[0].semilogy(hist['val'], label='val')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('MSE (scaled)'); ax[0].legend(); ax[0].set_title(f'{name} loss')
    for j, oname in enumerate(out_names):
        a = ax[j + 1]
        a.scatter(true[:, j], pred[:, j], s=8, alpha=0.5)
        lim = [min(true[:, j].min(), pred[:, j].min()), max(true[:, j].max(), pred[:, j].max())]
        a.plot(lim, lim, 'r--'); a.set_xlabel('true (clean)'); a.set_ylabel('NN prediction')
        a.set_title(f'{oname} test parity (R^2={metrics[j]["R2"]:.3f})')
    plt.tight_layout(); plt.savefig(f'{name}_results.png', dpi=150); plt.close()
    return model, (xm, xs, ym, ys), metrics
 
# ---------------- targets: name -> (clean input columns, noisy target(s), clean target(s)) ----------------
K_df = pd.DataFrame({'K_A': K_A, 'K_B': K_B, 'K_C': K_C})
K_df_clean = pd.DataFrame({'K_A': K_A_clean, 'K_B': K_B_clean, 'K_C': K_C_clean})

targets = {
    'theta_QR': (['VR', 'T1', 'TjR'], theta_QR, theta_QR_clean),
    'theta_QS': (['VS', 'TS', 'TjS'], theta_QS, theta_QS_clean),
    'r1':       (['CA1', 'T1'],       r1,       r1_clean),
    'r2':       (['CB1', 'T1'],       r2,       r2_clean),
    'K_ABC':    (['TS'],              K_df,     K_df_clean),      # 1 input -> 3 outputs
}

models, scalers, results = {}, {}, []
for name, (cols, y, y_clean) in targets.items():
    print(f"\n=== Training {name}({', '.join(cols)}) ===")
    m, sc, metrics = train_theta(clean[cols], y, y_clean, name)
    models[name], scalers[name] = m, sc
    results.extend(metrics)
    torch.save({'state_dict': m.state_dict(), 'inputs': cols,
                'outputs': list(y.columns) if isinstance(y, pd.DataFrame) else [name],
                'xm': sc[0], 'xs': sc[1], 'ym': sc[2], 'ys': sc[3]}, f'{name}_model.pt')
 
summary = pd.DataFrame(results).set_index('target')
print("\n=== Test-set summary ===")
print(summary)
summary.to_csv('nn_summary.csv')