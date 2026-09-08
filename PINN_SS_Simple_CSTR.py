import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class ReactionRateNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super(ReactionRateNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, C_A):
        return self.network(C_A)

def physics_informed_loss(r_pred, F_in, C_A0, F_out, C_A, V):
    mass_balance_residual = (F_in * C_A0) - (F_out * C_A) - (V * r_pred)
    return torch.mean(mass_balance_residual ** 2)

# Generate physically consistent data
# True rate law: r = 0.5 * C_A
torch.manual_seed(42)
batch_size = 200
V_val = 50.0
F_val = 2.0
k_true = 0.067  # True reaction rate constant

C_A0_data = torch.rand(batch_size, 1) * 10.0 +.01  # Inlet concentrations 0 to 10
# Steady state: F*C_A0 - F*C_A - V*(k*C_A) = 0
C_A_data = (F_val * C_A0_data) / (F_val + V_val * k_true)

F_in_data = torch.ones(batch_size, 1) * F_val
F_out_data = torch.ones(batch_size, 1) * F_val
V_data = torch.ones(batch_size, 1) * V_val

# Initialize model and optimizer
model = ReactionRateNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# add noise to the input data to make the model more robust
C_A_data_noisy = C_A_data + torch.randn_like(C_A_data) * 0.02


# Train
epochs = 20000
for epoch in range(epochs):
    optimizer.zero_grad()
    # put noise in the input data to make the model more robust
    # noise = torch.randn_like(C_A_data) * 0.02
    # C_A_data_noisy = C_A_data + noise
    r_pred = model(C_A_data_noisy)
    # add standard data 
    loss = physics_informed_loss(r_pred, F_in_data, C_A0_data, F_out_data, C_A_data_noisy, V_data)
    # loss = physics_informed_loss(r_pred, F_in_data, C_A0_data, F_out_data, C_A_data, V_data)
    loss.backward()
    optimizer.step()

# Plotting
# maybe switch C_A_data to C_A_data_noisy to see how the model performs with noisy data?
C_A_test = torch.linspace(0, C_A_data.max().item() * 1.2, 100).view(-1, 1)
r_pred_test = model(C_A_test).detach().numpy()
r_true_test = k_true * C_A_test.numpy()

plt.figure(figsize=(8, 6))
plt.plot(C_A_test.numpy(), r_true_test, 'k--', label='True Rate Law ($r = 2.0 C_A$)')
plt.plot(C_A_test.numpy(), r_pred_test, 'b-', label='Learned Rate Law (Neural Network)')
plt.scatter(C_A_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
plt.xlabel('Concentration of A ($C_A$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
plt.legend()
plt.grid(True)
plt.savefig('rate_plot.png')
print("Plot saved to rate_plot.png")