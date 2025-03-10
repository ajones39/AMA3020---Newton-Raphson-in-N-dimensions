import numpy as np 
import matplotlib.pyplot as plt

def F(X):
    x, y = X
    return np.array([np.exp(x) + y - 2, x**2 + y**2 - 4])

def J(X):
    x, y = X
    return np.array([[np.exp(x), 1], [2*x, 2*y]])

def newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=100):
    X = np.array(X0, dtype=float)
    iterates = [X.copy()]
    for i in range(max_iter):
        FX = F(X)
        delta_X = np.linalg.solve(J(X), -FX)
        X = X + delta_X
        iterates.append(X.copy())
        if np.linalg.norm(delta_X) < tol:
            break
    return X, iterates

def damped_newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=100, lambda_damping=0.1):
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
        if np.linalg.norm(delta_X) < tol:
            break
    return X, iterates

def compute_errors(iterates, root):
    return [np.linalg.norm(X - root) for X in iterates]

# Run 1: Initial guess [0.001, 0.001]
X0 = [0.001, 0.001]

# Undamped Newton-Raphson
root_undamped, iterates_undamped = newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=100)
errors_undamped = compute_errors(iterates_undamped, root_undamped)

# Damped Newton-Raphson
root_damped, iterates_damped = damped_newton_raphson_nd(F, J, X0, tol=1e-8, max_iter=100, lambda_damping=0.1)
errors_damped = compute_errors(iterates_damped, root_damped)

# Plots
plt.figure(figsize=(8,6))
plt.semilogy(errors_undamped, 'o-', label=f'Undamped: Initial guess {X0}, Converged to {np.round(root_undamped,4)}')
plt.semilogy(errors_damped, 's-', label=f'Damped: Initial guess {X0}, Converged to {np.round(root_damped,4)}')
plt.xlabel('Iteration number')
plt.ylabel('Error, Δ')
plt.title('Comparison of Convergence Rate: Undamped vs. Damped Newton-Raphson (2D)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(1e-14, 1e8)
plt.show()
