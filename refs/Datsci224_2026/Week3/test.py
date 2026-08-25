import numpy as np
import matplotlib.pyplot as plt


# Generate a grid of w and b values
w = np.linspace(-10, 10, 100)
b = np.linspace(-10, 10, 100)

W, B = np.meshgrid(w, b)

# Given our data point (1, 1), compute the loss for each combination of w and b
X = 1
Y_true = 1
Y_pred = W * X + B
MSE = (Y_true - Y_pred) ** 2

##########
# Plot Ridge regression
############
plt.clf()
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, MSE, cmap='viridis')

ax.set_xlabel('Coef')
ax.set_ylabel('Intercept')
ax.set_zlabel('Loss (MSE)')

plt.savefig("high_dim.png")

# Given our data point (1, 1), compute the ridge penalty
ridge = (W) ** 2
print("ridge SHAPE", ridge.shape)

plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, ridge, cmap='viridis')

ax.set_xlabel('Coef')
ax.set_ylabel('Intercept')
ax.set_zlabel('Ridge penalty')

plt.savefig("high_dim_ridge.png")

# plot ridge regression loss
ridge_loss = MSE + ridge

plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, ridge_loss, cmap='viridis')

ax.set_xlabel('Coef')
ax.set_ylabel('Intercept')
ax.set_zlabel('Ridge penalized loss')

plt.savefig("high_dim_ridge_loss.png")


##########
# Plot lasso regression
############
lasso = np.abs(W)
lasso_loss = MSE + lasso * 40

plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, lasso, cmap='viridis')

ax.set_xlabel('Coef')
ax.set_ylabel('Intercept')
ax.set_zlabel('Lasso penalty')

plt.show()
# plt.savefig("high_dim_lasso.png")


plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, lasso_loss, cmap='viridis')

ax.set_xlabel('Coef')
ax.set_ylabel('Intercept')
ax.set_zlabel('Lasso penalized loss')
# plt.show()
# plt.savefig("high_dim_lasso_loss.png")