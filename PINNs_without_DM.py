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