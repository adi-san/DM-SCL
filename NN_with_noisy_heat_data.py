import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

torch.manual_seed(0)
np.random.seed(0)

# read the noisy heat dm CSV

data = pd.read_csv('noisy_heat_DM.csv')
print(data)

# learn theta QR as a function of VR, T1, and TjR
# learn theta QS as a function of VS, TS, and TjS
# data is contained in the data df, use the 125 inputs and outputs to train the NNs
# Test using a grid of points spanning from the minimum and maximum of the measured parameters --> generate a parity plot comparing NN to actual

# ---------------- NN learning ----------------
UaR, UaS = 5000.0, 4000.0   # true heat transfer coefficients (from traditional_GPs.py) -> "actual" on the grid
true_fns = {
    'theta_QR': lambda X: UaR * X[:, 0] * (X[:, 1] - X[:, 2]),   # inputs: VR, T1, TjR
    'theta_QS': lambda X: UaS * X[:, 0] * (X[:, 2] - X[:, 1]),   # inputs: VS, TS, TjS
}
tasks = {
    'theta_QR': ['VR', 'T1', 'TjR'],
    'theta_QS': ['VS', 'TS', 'TjS'],
}
N_GRID = 5   # points per input dimension -> 5^3 = 125 test points
HIDDEN = 32
EPOCHS = 2000
LR = 1e-2
WEIGHT_DECAY = 1e-3   # L2 regularization plays the role of the GP's WhiteKernel: keeps the NN from fitting the noise


def make_net(d):
    return nn.Sequential(nn.Linear(d, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, 1))


def fit_net(Xn, yn):
    net = make_net(Xn.shape[1])
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(net(Xn), yn)
        loss.backward()
        opt.step()
        sched.step()
    return net, loss.item()


def predict(net, X, m, s, a, b):
    with torch.no_grad():
        out = net(torch.tensor((X - m) / s, dtype=torch.float32)).numpy().ravel()
    return out * b + a


fig, axes = plt.subplots(2, len(tasks), figsize=(6 * len(tasks), 11))
for col, (target, inputs) in enumerate(tasks.items()):
    ax, ax_cv = axes[0, col], axes[1, col]
    X = data[inputs].values.astype(float)     # 125 training inputs
    y = data[target].values.astype(float)     # 125 noisy training outputs

    # standardize inputs and output
    xm, xs = X.mean(0), X.std(0)
    ym, ys = y.mean(), y.std()
    Xn = torch.tensor((X - xm) / xs, dtype=torch.float32)
    yn = torch.tensor(((y - ym) / ys)[:, None], dtype=torch.float32)
    net, train_mse = fit_net(Xn, yn)
    print(f"\n{target}: final training MSE (standardized) = {train_mse:.4f}")

    # test grid spanning min..max of each measured input
    axes_1d = [np.linspace(X[:, j].min(), X[:, j].max(), N_GRID) for j in range(X.shape[1])]
    Xg = np.stack([m.ravel() for m in np.meshgrid(*axes_1d, indexing='ij')], axis=1)
    y_true = true_fns[target](Xg)
    y_pred = predict(net, Xg, xm, xs, ym, ys)

    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"{target}: grid R^2 = {r2:.4f}, RMSE = {rmse:.4g} ({100 * rmse / y_true.std():.2f}% of std)")

    # parity plot: NN vs actual
    ax.plot(y_true, y_pred, 'o', ms=3, alpha=0.4, label='NN')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', label='parity')
    ax.set_xlabel(f'actual {target}')
    ax.set_ylabel(f'NN predicted {target}')
    ax.set_title(f'{target}  (R² = {r2:.4f})')
    ax.legend()

    # 5-fold cross-validation on the 125 measured points: NN vs measured (held-out) values
    y_cv = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m_, s_ = X[tr].mean(0), X[tr].std(0)
        a_, b_ = y[tr].mean(), y[tr].std()
        Xt = torch.tensor((X[tr] - m_) / s_, dtype=torch.float32)
        yt = torch.tensor(((y[tr] - a_) / b_)[:, None], dtype=torch.float32)
        g, _ = fit_net(Xt, yt)
        y_cv[te] = predict(g, X[te], m_, s_, a_, b_)
    r2_cv = 1 - np.sum((y - y_cv) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{target}: 5-fold CV R^2 vs measured = {r2_cv:.4f}")
    ax_cv.plot(y, y_cv, 'o', ms=4, alpha=0.7)
    ax_cv.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax_cv.set_xlabel(f'measured {target}')
    ax_cv.set_ylabel(f'NN predicted {target} (held-out)')
    ax_cv.set_title(f'{target} 5-fold CV  (R² = {r2_cv:.4f})')

    torch.save({'state_dict': net.state_dict(), 'xm': xm, 'xs': xs, 'ym': ym, 'ys': ys,
                'inputs': inputs}, f'nn_noisy_heat_{target}.pt')

plt.tight_layout()
plt.savefig('nn_noisy_heat_parity.png', dpi=150)
print("\nSaved nn_noisy_heat_parity.png")
