import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class ReactionRateNN(nn.Module):
    def __init__(self, hidden_dim=7):
        super(ReactionRateNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, C):
        # x = torch.cat([C, F_out], dim=1)
        return self.network(C)
class ReactionRateNN_C_A_C_B(nn.Module):
    def __init__(self, hidden_dim=4):
        super(ReactionRateNN_C_A_C_B, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, C_A, C_B):
        x = torch.cat([C_A, C_B], dim=1)
        return self.network(x)

def physics_informed_loss_C_A(r_pred, F_in, C_A0, F_out, C_A, V):
    mass_balance_residual = (F_in * C_A0) - (F_out * C_A) - (V * r_pred)
    return torch.mean(mass_balance_residual ** 2)
def physics_informed_loss_C_B(r_pred, F_out, C_B, V):
    mass_balance_residual = -(F_out * C_B) + (V * r_pred)
    return torch.mean(mass_balance_residual ** 2)
def noise_term(tensor, noise_std):
    noise = torch.randn_like(tensor) * noise_std
    return noise

# Generate physically consistent data
# True rate law: r = k_true * C_A
torch.manual_seed(42)
batch_size = 100
V_val = 50.0
F_val = 3.0
k_true = 0.1  # True reaction rate constant

# C_A0_data = torch.rand(batch_size, 1) * 20.0 + 0.01  # Inlet concentrations 0 to 20
# make C_A0_data linearly spaced instead of random
C_A0_data = torch.linspace(0.01, 40.0, batch_size).view(-1, 1)
# Steady state: F*C_A0 - F*C_A - V*(k*C_A) = 0
C_A_data = (F_val * C_A0_data) / (F_val + V_val * k_true)
std_C_A_data=torch.std(C_A_data)
# Want to also compare the Ca input to the Cb input, so we can generate Cb data as well
C_B_data = C_A0_data-C_A_data  # Assuming no B is present in the inlet
std_C_B_data=torch.std(C_B_data)
F_in_data = torch.ones(batch_size, 1) * F_val
F_out_data = torch.ones(batch_size, 1) * F_val
V_data = torch.ones(batch_size, 1) * V_val

# Initialize model and optimizer
model = ReactionRateNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# the noise level for conc is set at 0.2, which is 20% of the standard deviation of the data.
conc_noise_level=0.25

# add noise to the input data to make the model more robust
C_A_data_noisy = C_A_data + torch.randn_like(C_A_data) * conc_noise_level*std_C_A_data
# trim the noisy point that are negative, since concentration cannot be negative
C_A_data_noisy = torch.clamp(C_A_data_noisy, min=0.0)
# Cb data is not used in the loss function, but we can still add noise to it for completeness
C_B_data_noisy = C_B_data + torch.randn_like(C_B_data) * conc_noise_level*std_C_B_data
# trim the noisy point that are negative, since concentration cannot be negative
C_B_data_noisy = torch.clamp(C_B_data_noisy, min=0.0)

# what if I added noise to F_out data? 
F_out_data_noisy = F_out_data 
# + torch.randn_like(F_out_data) * 0.01

# Train
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    # put noise in the input data to make the model more robust
    # noise = torch.randn_like(C_A_data) * 0.02
    # C_A_data_noisy = C_A_data + noise
    r_pred = model(C_A_data_noisy)
    # add standard data 
    loss = physics_informed_loss_C_A(r_pred, F_in_data, C_A0_data, F_out_data, C_A_data_noisy, V_data)
    # loss = physics_informed_loss(r_pred, F_in_data, C_A0_data, F_out_data, C_A_data, V_data)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Initialize model and optimizer C_B
model_C_B = ReactionRateNN()
optimizer_C_B = optim.Adam(model_C_B.parameters(), lr=0.001)

print(C_A_data_noisy.shape)    # Expected: torch.Size([100, 1])
print(F_out_data_noisy.shape)  # Expected: torch.Size([100, 1])
print(C_B_data_noisy.shape)    # Expected: torch.Size([100, 1])

for epoch in range(epochs):
    optimizer_C_B.zero_grad()
    r_pred_C_B = model_C_B(C_B_data_noisy)
    loss_C_B = physics_informed_loss_C_B(r_pred_C_B, F_out_data, C_B_data_noisy, V_data)
    loss_C_B.backward()
    optimizer_C_B.step()
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss C_B: {loss_C_B.item():.6f}')

# Plotting
# maybe switch C_A_data to C_A_data_noisy to see how the model performs with noisy data?

# length of test data is 100, so we can use that to generate the test data
# we also want to see whether test data is somewhat outside the range of the training data, so we can use 1.2 times the max of the training data as the upper limit for the test data
C_A_test = torch.linspace(0, C_A_data.max().item() * 1.5, 100).view(-1, 1)
F_out_data_test = torch.ones_like(C_A_test) * F_val
r_pred_test = model(C_A_test).detach().numpy()
r_true_test = k_true * C_A_test.numpy()

plt.figure(figsize=(8, 6))
plt.plot(C_A_test.numpy(), r_true_test, 'k--', label='True Rate Law ($r = k C_A$)')
plt.plot(C_A_test.numpy(), r_pred_test, 'b-', label='Learned Rate Law (Neural Network)')
plt.scatter(C_A_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
plt.xlabel('Concentration of A ($C_A$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
plt.legend()
plt.grid(True)
plt.show()
print("Plot saved to rate_plot.png")

# plotting for C_B

C_B_test = torch.linspace(0, C_B_data.max().item() * 1.5, 100).view(-1, 1)
r_pred_test_C_B = model_C_B(C_B_test).detach().numpy()

plt.figure(figsize=(8, 6))
plt.plot(C_B_test.numpy(), r_true_test, 'k--', label='True Rate Law ($r = k C_A$)')
plt.plot(C_B_test.numpy(), r_pred_test_C_B, 'b-', label='Learned Rate Law (Neural Network)')
plt.scatter(C_B_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
plt.xlabel('Concentration of B ($C_B$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
plt.legend()
plt.grid(True)
plt.show()
print("Plot saved to rate_plot.png")

# collect the weights and biases of each model and save them to a csv file for later use
torch.save(model.state_dict(), 'model_C_A.pth')
torch.save(model_C_B.state_dict(), 'model_C_B.pth')
