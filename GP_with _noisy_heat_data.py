import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.model_selection import KFold


# read the noisy heat dm CSV

data=pd.read_csv('noisy_heat_DM.csv')
print(data)

# learn theta QR as a function of VR, T1, and TjR
# learn theta QS as a function of VS, TS, and TjS
# data is contained in the data df, use the 125 inputs and outputs to train the GPs
# Test using a grid of points spanning from the minimum and maximum of the measured parameters --> generate a parity plot comparing GP to actual

# ---------------- GP learning ----------------
UaR, UaS = 5000.0, 4000.0   # true heat transfer coefficients (from traditional_GPs.py) -> "actual" on the grid
true_fns = {
    'theta_QR': lambda X: UaR * X[:, 0] * (X[:, 1] - X[:, 2]),   # inputs: VR, T1, TjR
    'theta_QS': lambda X: UaS * X[:, 0] * (X[:, 2] - X[:, 1]),   # inputs: VS, TS, TjS
}
tasks = {
    'theta_QR': ['VR', 'T1', 'TjR'],
    'theta_QS': ['VS', 'TS', 'TjS'],
}
N_GRID = 5   # points per input dimension -> 10^3 = 1000 test points


def make_kernel(d):
    # ARD Matern-5/2 (one lengthscale per input) + WhiteKernel to learn the measurement noise
    return (C(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-2, (1e-8, 1.0)))


fig, axes = plt.subplots(2, len(tasks), figsize=(6 * len(tasks), 11))
for col, (target, inputs) in enumerate(tasks.items()):
    ax, ax_cv = axes[0, col], axes[1, col]
    X = data[inputs].values.astype(float)     # 125 training inputs
    y = data[target].values.astype(float)     # 125 noisy training outputs

    # standardize inputs and output
    xm, xs = X.mean(0), X.std(0)
    ym, ys = y.mean(), y.std()
    gp = GPR(make_kernel(X.shape[1]), alpha=1e-8, n_restarts_optimizer=10, random_state=0)
    gp.fit((X - xm) / xs, (y - ym) / ys)
    print(f"\n{target}: learned kernel = {gp.kernel_}")

    # test grid spanning min..max of each measured input
    axes_1d = [np.linspace(X[:, j].min(), X[:, j].max(), N_GRID) for j in range(X.shape[1])]
    Xg = np.stack([m.ravel() for m in np.meshgrid(*axes_1d, indexing='ij')], axis=1)
    y_true = true_fns[target](Xg)
    mu, sd = gp.predict((Xg - xm) / xs, return_std=True)
    y_pred, y_std = mu * ys + ym, sd * ys

    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"{target}: grid R^2 = {r2:.4f}, RMSE = {rmse:.4g} ({100 * rmse / y_true.std():.2f}% of std)")

    # parity plot: GP vs actual
    ax.errorbar(y_true, y_pred, yerr=2 * y_std, fmt='o', ms=3, alpha=0.4, elinewidth=0.5, label='GP mean Â± 2Ïƒ')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', label='parity')
    ax.set_xlabel(f'actual {target}')
    ax.set_ylabel(f'GP predicted {target}')
    ax.set_title(f'{target}  (RÂ² = {r2:.4f})')
    ax.legend()

    # 5-fold cross-validation on the 125 measured points: GP vs measured (held-out) values
    y_cv = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m_, s_ = X[tr].mean(0), X[tr].std(0)
        a_, b_ = y[tr].mean(), y[tr].std()
        g = GPR(make_kernel(X.shape[1]), alpha=1e-8, n_restarts_optimizer=3, random_state=0)
        g.fit((X[tr] - m_) / s_, (y[tr] - a_) / b_)
        y_cv[te] = g.predict((X[te] - m_) / s_) * b_ + a_
    r2_cv = 1 - np.sum((y - y_cv) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{target}: 5-fold CV R^2 vs measured = {r2_cv:.4f}")
    ax_cv.plot(y, y_cv, 'o', ms=4, alpha=0.7)
    ax_cv.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax_cv.set_xlabel(f'measured {target}')
    ax_cv.set_ylabel(f'GP predicted {target} (held-out)')
    ax_cv.set_title(f'{target} 5-fold CV  (R² = {r2_cv:.4f})')

    joblib.dump({'gp': gp, 'xm': xm, 'xs': xs, 'ym': ym, 'ys': ys, 'inputs': inputs},
                f'gp_noisy_heat_{target}.joblib')

plt.tight_layout()
plt.savefig('gp_noisy_heat_parity.png', dpi=150)
print("\nSaved gp_noisy_heat_parity.png")
