import numpy as np
import matplotlib.pyplot as plt

# Parameters
true_k = 5.5
true_b = -1.0
n_samples = 100
noise_std = 2.0
learning_rate = 0.01
n_iter = 200

print("true_k =", true_k)
print("true_b =", true_b)
print("n_samples =", n_samples)
print("noise_std =", noise_std)
print("learning_rate =", learning_rate)
print("n_iter =", n_iter)

# Data generation
np.random.seed(42)
x = np.random.rand(n_samples) * 10
y = true_k * x + true_b + np.random.randn(n_samples) * noise_std

# Task 1: Least Squares
def least_squares(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    k = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    b = y_mean - k * x_mean
    return k, b

k_ls, b_ls = least_squares(x, y)
k_poly, b_poly = np.polyfit(x, y, 1)

print("\nMethod of Least Squares:")
print("  k_ls =", k_ls, " b_ls =", b_ls)
print("NumPy polyfit:")
print("  k_poly =", k_poly, " b_poly =", b_poly)

plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, true_k*x + true_b, color='black', linewidth=2, label='True line')
plt.plot(x, k_ls*x + b_ls, linestyle='--', label='Least Squares')
plt.plot(x, k_poly*x + b_poly, linestyle=':', label='np.polyfit')
plt.legend()
plt.title("Task 1: Regression Lines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Task 2: Gradient Descent
def gradient_descent(x, y, lr, n_iter):
    b = 0.0
    k = 0.0
    n = len(x)
    history = []
    for i in range(n_iter):
        y_pred = k * x + b
        error = y - y_pred
        grad_b = -2/n * np.sum(error)
        grad_k = -2/n * np.sum(x * error)
        b -= lr * grad_b
        k -= lr * grad_k
        mse = np.mean(error**2)
        history.append(mse)
    return k, b, history

k_gd, b_gd, history = gradient_descent(x, y, learning_rate, n_iter)

print("\nGradient Descent:")
print("  k_gd =", k_gd, " b_gd =", b_gd)

plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, true_k*x + true_b, color='black', linewidth=2, label='True line')
plt.plot(x, k_ls*x + b_ls, linestyle='--', label='Least Squares')
plt.plot(x, k_gd*x + b_gd, linestyle='-.', label='Gradient Descent')
plt.legend()
plt.title("Task 2: Gradient Descent Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure()
plt.plot(range(1, n_iter+1), history)
plt.title("MSE vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.show()
