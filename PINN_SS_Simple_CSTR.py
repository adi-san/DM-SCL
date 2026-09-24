import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

class ReactionRateNN(nn.Module):
    def __init__(self, hidden_dim=6):
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
class ReactionRateNN_C_B(nn.Module):
    def __init__(self, hidden_dim=6):
        super(ReactionRateNN_C_B, self).__init__()
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
def physics_informed_loss_z(r_pred, F_out, z, C_A0, V):
    z_residual = F_out *(z-C_A0) + (V * r_pred)
    return torch.mean(z_residual ** 2) 
def noise_term(tensor, noise_std):
    noise = torch.randn_like(tensor) * noise_std
    return noise

# Generate physically consistent data
# True rate law: r = k_true * C_A
torch.manual_seed(42)
batch_size = 30
V_val = 50.0
F_val = 3.0
k_true = 0.1  # True reaction rate constant

# C_A0_data = torch.rand(batch_size, 1) * 20.0 + 0.01  # Inlet concentrations 0 to 20
# make C_A0_data linearly spaced instead of random
C_A0_data = torch.linspace(0.01, 20.0, batch_size).view(-1, 1)
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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# the noise level for conc is set at 0.2, which is 20% of the standard deviation of the data.
conc_noise_level=0.3

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
model_C_B = ReactionRateNN_C_B()
optimizer_C_B = optim.Adam(model_C_B.parameters(), lr=0.001)

print(C_A_data_noisy.shape)    # Expected: torch.Size([100, 1])
print(F_out_data_noisy.shape)  # Expected: torch.Size([100, 1])
print(C_B_data_noisy.shape)    # Expected: torch.Size([100, 1])

for epoch in range(epochs):
    optimizer_C_B.zero_grad()
    r_pred_C_B = model_C_B(C_A0_data,C_B_data_noisy)
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
# Ensure that my test data for C_A0 is larger than C_B_test for all points
# for each C_B_test point, set C_A0_test to be randomly drawn from a uniform distribution between C_B_test and 1.5 times the max of C_A0_data
C_A0_test = torch.zeros_like(C_B_test)
for i in range(C_B_test.shape[0]):
    C_A0_test[i] = torch.rand(1) * (C_A0_data.max().item() * 1.5 - C_B_test[i]) + C_B_test[i]

# C_A0_test = torch.ones_like(C_B_test) * C_A0_data.max().item() * 1.5
r_pred_test_C_B = model_C_B(C_A0_test,C_B_test).detach().numpy()

r_true_test_C_B = k_true * (C_A0_test-C_B_test).numpy()
"""
plt.figure(figsize=(8, 6))
plt.plot(C_A0_test.numpy()-C_B_test.numpy(), r_true_test_C_B, 'k--', label='True Rate Law ($r = k C_A$)')
plt.plot(C_A0_test.numpy()-C_B_test.numpy(), r_pred_test_C_B, 'b-', label='Learned Rate Law (Neural Network)')
plt.scatter(C_A0_data.numpy()-C_B_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
plt.xlabel('Concentration of C_A0 Minus C_B ($C_{A0}$-$C_B$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
plt.legend()
plt.grid(True)
plt.show()
print("Plot saved to rate_plot.png")
"""

# 1. Calculate the x-axis values (C_A0 - C_B) and flatten them to 1D arrays
x_test = (C_A0_test - C_B_test).numpy().flatten()
r_true_test_flat = r_true_test_C_B.flatten()
r_pred_test_flat = r_pred_test_C_B.flatten()

# 2. Get the indices that would sort the x-axis values
sort_idx = np.argsort(x_test)

# 3. Apply the sorting indices to x and both y arrays
x_test_sorted = x_test[sort_idx]
r_true_sorted = r_true_test_flat[sort_idx]
r_pred_sorted = r_pred_test_flat[sort_idx]

# 4. Plot using the sorted data
plt.figure(figsize=(8, 6))

plt.plot(x_test_sorted, r_true_sorted, 'k--', label='True Rate Law ($r = k C_A$)')
plt.plot(x_test_sorted, r_pred_sorted, 'b-', label='Learned Rate Law (Neural Network)')

# Note: Keeping your original scatter plot for the training data
plt.scatter(C_A0_data.numpy() - C_B_data_noisy.numpy(), (k_true * C_A_data).numpy(), 
            color='red', alpha=0.5, label='Training Data Extent')

# Cleaned up x-axis label with proper LaTeX subscript
plt.xlabel('Concentration Difference ($C_{A0} - C_B$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
plt.legend()
plt.grid(True)
plt.show()
# want a 3d plot with C_A0 and C_B on the x and y axes, and r on the z axis

# # 1. Create a 2D grid (meshgrid) for C_A0 and C_B
# grid_resolution = 50
# C_A0_vals = torch.linspace(0, C_A0_data.max().item() * 1.5, grid_resolution)
# C_B_vals = torch.linspace(0, C_B_data.max().item() * 1.5, grid_resolution)

# C_A0_mesh, C_B_mesh = torch.meshgrid(C_A0_vals, C_B_vals, indexing='ij')

# # 2. Flatten the grids to pass into the model
# C_A0_flat = C_A0_mesh.reshape(-1, 1)
# C_B_flat = C_B_mesh.reshape(-1, 1)

# # 3. Get model predictions and true values
# r_pred_flat = model_C_B(C_A0_flat, C_B_flat).detach().numpy()
# r_true_flat = k_true * (C_A0_flat - C_B_flat).numpy()

# # 4. Reshape back to 2D for surface plotting
# r_pred_mesh = r_pred_flat.reshape(C_A0_mesh.shape)
# r_true_mesh = r_true_flat.reshape(C_A0_mesh.shape)

# # 5. Mask out unphysical regions where C_B > C_A0
# unphysical_mask = (C_B_mesh > C_A0_mesh).numpy()
# r_pred_mesh[unphysical_mask] = np.nan
# r_true_mesh[unphysical_mask] = np.nan

# # 6. Generate the 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the True Rate Law (as a semi-transparent gray surface)
# ax.plot_surface(C_A0_mesh.numpy(), C_B_mesh.numpy(), r_true_mesh, 
#                 color='gray', alpha=0.4, antialiased=True)

# # Plot the Learned PINN Rate Law (as a colormapped surface)
# surf_pred = ax.plot_surface(C_A0_mesh.numpy(), C_B_mesh.numpy(), r_pred_mesh, 
#                             cmap='viridis', alpha=0.8, antialiased=True)

# # Proxy artists for the legend (since 3D surfaces don't map to legends natively very well)
# true_patch = mpatches.Patch(color='gray', alpha=0.4, label='True Rate Law ($r = k (C_{A0} - C_B)$)')
# pred_patch = mpatches.Patch(color='teal', alpha=0.8, label='Learned Rate Law (PINN)')
# ax.legend(handles=[true_patch, pred_patch], loc='upper left')

# # Labels and title
# ax.set_xlabel('Initial Concentration ($C_{A0}$)')
# ax.set_ylabel('Concentration of B ($C_B$)')
# ax.set_zlabel('Reaction Rate ($r$)')
# ax.set_title('3D Surface: Learned vs. True Reaction Rate')

# # Optional: Adjust viewing angle for better perspective
# ax.view_init(elev=25, azim=40) 

# plt.show()

# define z as C_A0 - C_B, we will train a NN feeding z
z_train = C_A0_data - C_B_data_noisy
# z should be strictly positive, so we can clamp it to be at least 0.01
z_train = torch.clamp(z_train, min=0.01)

model_z= ReactionRateNN()
optimizer_z = optim.Adam(model_z.parameters(), lr=0.0001)

for epoch in range(epochs):
    optimizer_z.zero_grad()
    r_pred_z = model_z(z_train)
    loss_z = physics_informed_loss_z(r_pred_z, F_out_data, z_train, C_A0_data, V_data)
    loss_z.backward()
    optimizer_z.step()
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss C_B: {loss_z.item():.6f}')

z_test = torch.linspace(0, 10, 100).view(-1, 1)
r_pred_test_z = model_z(z_test).detach().numpy()
r_true_test_z = k_true * z_test.numpy()
plt.figure(figsize=(8, 6))
plt.plot(z_test.numpy(), r_true_test_z, 'k--', label='True Rate Law ($r = k z$)')
plt.plot(z_test.numpy(), r_pred_test_z, 'b-', label='Learned Rate Law (Neural Network)')
plt.scatter(z_train.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
plt.xlabel('Concentration Difference ($z = C_{A0} - C_B$)')
plt.ylabel('Reaction Rate ($r$)')
plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate (z)')
plt.legend()
plt.grid(True)
plt.show()
# C_A_test = torch.linspace(0, C_A_data.max().item() * 1.5, 100).view(-1, 1)
# F_out_data_test = torch.ones_like(C_A_test) * F_val
# r_pred_test = model(C_A_test).detach().numpy()
# r_true_test = k_true * C_A_test.numpy()

# plt.figure(figsize=(8, 6))
# plt.plot(C_A_test.numpy(), r_true_test, 'k--', label='True Rate Law ($r = k C_A$)')
# plt.plot(C_A_test.numpy(), r_pred_test, 'b-', label='Learned Rate Law (Neural Network)')
# plt.scatter(C_A_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data Extent')
# plt.xlabel('Concentration of A ($C_A$)')
# plt.ylabel('Reaction Rate ($r$)')
# plt.title('Physics-Informed Neural Network: Learned vs. True Reaction Rate')
# plt.legend()
# plt.grid(True)
# plt.show()
# print("Plot saved to rate_plot.png")


# # need to make a 3d plot with C_A0 and C_B on the x and y axes, and r on the z axis
# # form a mesh of points for C_A0 and C_B such that C_A0 is always greater than C_B, since C_B cannot be greater than C_A0
# C_A0_mesh, C_B_mesh = np.meshgrid(np.linspace(0, C_A0_data.max().item() * 1.5, 50), np.linspace(0, C_B_data.max().item() * 1.5, 50))
# # flatten the mesh to pass through the model
# C_A0_mesh_flat = torch.tensor(C_A0_mesh.flatten(), dtype=torch.float32).view(-1, 1)
# C_B_mesh_flat = torch.tensor(C_B_mesh.flatten(), dtype=torch.float32).view(-1, 1)
# # filter out points where C_B > C_A0, since that is not physically possible
# mask = C_B_mesh_flat <= C_A0_mesh_flat
# C_A0_mesh_flat = C_A0_mesh_flat[mask]
# C_B_mesh_flat = C_B_mesh_flat[mask]
# # get the predicted reaction rates for the mesh points
# r_pred_mesh = model_C_B(C_A0_mesh_flat, C_B_mesh_flat).detach().numpy()
# # reshape the predicted rates back to the mesh shape
# r_pred_mesh_full = np.full(C_A0_mesh.shape, np.nan)
# #plot the predicted rates on the mesh, but only for the valid points
# r_pred_mesh_full[mask.numpy().reshape(C_A0_mesh.shape)] = r_pred_mesh
# plt.figure(figsize=(10, 8))
# ax = plt.axes(projection='3d')
# ax.plot_surface(C_A0_mesh, C_B_mesh, r_pred_mesh_full, cmap='viridis', edgecolor='none')
# ax.set_xlabel('Concentration of A ($C_{A0}$)')
# ax.set_ylabel('Concentration of B ($C_B$)')
# ax.set_zlabel('Reaction Rate ($r$)')
# ax.set_title('Learned Reaction Rate Surface')
# #show the training data points on the surface plot
# ax.scatter(C_A0_data.numpy(), C_B_data_noisy.numpy(), (k_true * C_A_data).numpy(), color='red', alpha=0.5, label='Training Data')
# plt.legend()
# # collect the weights and biases of each model and save them to a csv file for later use
# torch.save(model.state_dict(), 'model_C_A.pth')
# torch.save(model_C_B.state_dict(), 'model_C_B.pth')
