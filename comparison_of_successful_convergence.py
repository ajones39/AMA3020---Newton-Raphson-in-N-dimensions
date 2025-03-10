
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

def compute_errors(iterates, root):
    return [np.linalg.norm(X - root) for X in iterates]

# Run 1: Good initial guess [1, -2]
X0_1 = [1, -2]
root1, iterates1 = newton_raphson_nd(F, J, X0_1, tol=1e-8, max_iter=100)
errors1 = compute_errors(iterates1, root1)

# Run 2: Alternative initial guess [2, 10]
X0_2 = [2, 10]
root2, iterates2 = newton_raphson_nd(F, J, X0_2, tol=1e-8, max_iter=100)
errors2 = compute_errors(iterates2, root2)

# Run 3: Poor initial guess [10, 10]
X0_3 = [10, 10]
root3, iterates3 = newton_raphson_nd(F, J, X0_3, tol=1e-8, max_iter=100)
errors3 = compute_errors(iterates3, root3)

# Calculate reference quadratic error for Run 1 (good guess)
ref_errors1 = [errors1[0]]
if errors1[0] != 0 and len(errors1) > 1:
    A1 = errors1[1] / (errors1[0]**2)
    for k in range(1, len(errors1)):
        next_err1 = A1 * (ref_errors1[-1])**2
        ref_errors1.append(next_err1)
else:
    ref_errors1 = errors1  

ref_errors3 = [errors3[0]]
if errors3[0] != 0 and len(errors3) > 1:
    A3 = errors3[1] / (errors3[0]**2)
    for k in range(1, len(errors3)):
        next_err3 = A3 * (ref_errors3[-1])**2
        ref_errors3.append(next_err3)
else:
    ref_errors3 = errors3  

#Plots
plt.figure(figsize=(8,6))
plt.semilogy(errors1, 'o-', label=f'Initial guess {X0_1}, Converged to {np.round(root1,4)}')
plt.semilogy(errors2, 's-', label=f'Initial guess {X0_2}, Converged to {np.round(root2,4)}')
plt.semilogy(errors3, 'd-', label=f'Initial guess {X0_3}, Converged to {np.round(root3,4)}')
plt.semilogy(ref_errors1, 'k--', linewidth=2, label='Reference quadratic error for good guess at (1, -2)')
plt.semilogy(ref_errors3, 'c--', linewidth=2, label='Reference quadratic error for poor guess at (10,10)')
plt.xlabel('Iteration number')
plt.ylabel('Error, Δ')
plt.title('Rate of Convergence for Newton-Raphson in 2D')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(1e-14, 1e2)
plt.show()
