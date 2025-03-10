
import numpy as np
import matplotlib.pyplot as plt

def F(X):
    x, y = X
    return np.array([np.exp(x) + y - 2, x**2 + y**2 - 4])

def J(X):
    x, y = X
    return np.array([[np.exp(x), 1], [2*x, 2*y]])

def damped_newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=20, lambda_damping=1):
    X = np.array(X0, dtype=float)
    iterates = [X.copy()]
    for i in range(max_iter):
        FX = F(X)
        jacobian = J(X)
        
        # Damping term added to the Jacobian
        J_damped = jacobian + lambda_damping * np.eye(len(X))
        
        try:
            delta_X = np.linalg.solve(J_damped, -FX)
        except np.linalg.LinAlgError:
            print(f"Jacobian is singular at iteration {i} with X = {X}")
            break
        
        X += delta_X
        iterates.append(X.copy())
        print(f"Iteration {i+1}: X = {X}, ||delta_X|| = {np.linalg.norm(delta_X)}")
        
        if np.linalg.norm(delta_X) < tol:
            print("Convergence achieved.")
            break
    return X, iterates

# Use a poor initial guess that is far from any root.
X0 = [0.001, 0.001]
root, iterates = damped_newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=20, lambda_damping=0.1)
print(f"Final iterate: {root}")

# Prepare grid for contour plots
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z1 = np.exp(X_grid) + Y_grid - 2
Z2 = X_grid**2 + Y_grid**2 - 4

plt.figure(figsize=(8,6))
contour1 = plt.contour(X_grid, Y_grid, Z1, levels=[0], colors='blue', linestyles='dashed')
contour2 = plt.contour(X_grid, Y_grid, Z2, levels=[0], colors='red', linestyles='solid')
plt.clabel(contour1, fmt='F1=0', colors='blue')
plt.clabel(contour2, fmt='F2=0', colors='red')

iterates = np.array(iterates)
plt.plot(iterates[:,0], iterates[:,1], 'ko-', label='Damped Newton iterations')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Damped Newton–Raphson Iterations for the 2D system')
plt.legend()
plt.show()
