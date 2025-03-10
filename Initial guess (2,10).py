import numpy as np
import matplotlib.pyplot as plt

def F(X):
    x, y = X
    return np.array([np.exp(x) + y - 2, x**2 + y**2 - 4])

def J(X):
    x, y = X
    return np.array([[np.exp(x), 1], [2*x, 2*y]])

def newton_raphson_nd(F, J, X0, tol=1e-20, max_iter=100):
    X = np.array(X0, dtype=float)
    iterates = [X.copy()]
    for i in range(max_iter):
        FX = F(X)

        if np.linalg.norm(FX) < tol:
            break
        delta_X = np.linalg.solve(J(X), -FX)  # Solve J(X)*delta_X = -F(X)
        X += delta_X
        iterates.append(X.copy())

        if np.linalg.norm(delta_X) < tol:
            break
    return X, iterates

# Initial guess
X0 = [2, 10]
root, iterates = newton_raphson_nd(F, J, X0)
print(f"Root found: {root} after {len(iterates)-1} iterations")

# contour plots
x_vals = np.linspace(-2, 3, 1000)
y_vals = np.linspace(-2, 10, 1000)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z1 = np.exp(X_grid) + Y_grid - 2
Z2 = X_grid**2 + Y_grid**2 - 4

# Plotting the contours and the iterates
plt.figure(figsize=(8,6))
contour1 = plt.contour(X_grid, Y_grid, Z1, levels=[0], colors='blue', linestyles='dashed')
contour2 = plt.contour(X_grid, Y_grid, Z2, levels=[0], colors='red', linestyles='solid')
plt.clabel(contour1, fmt='F1=0', colors='blue')
plt.clabel(contour2, fmt='F2=0', colors='red')

# Convert iterates list to a NumPy array for easier plotting
iterates_arr = np.array(iterates)
plt.plot(iterates_arr[:, 0], iterates_arr[:, 1], 'ko-', label='Newton iterations')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton-Raphson Iterations for a 2D System')
plt.legend()
plt.show()
