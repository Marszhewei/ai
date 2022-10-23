#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt


def load_data():
    print("data loading...")
    data = np.loadtxt("data.csv", delimiter=",", dtype=np.float64)
    y = data[:, -1]
    y = y.reshape(-1, 1)
    x = data[:, 0:-1]
    x1 = x.copy()
    x, mu, sigma = feature_normaliza(x)
    x = np.hstack((np.ones((len(y), 1)), x))
    x1 = np.hstack((np.ones((len(y), 1)), x1))
    theta = np.zeros((data.shape[1], 1))
    print("data loaded")
    return x, x1, y, theta, mu, sigma


def feature_normaliza(x):
    mu = np.zeros((1, x.shape[1]))
    sigma = np.zeros((1, x.shape[1]))
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - mu[i]) / sigma[i]
    return x, mu, sigma


def gradient_descent(x, y, theta, alpha, iters_n):
    m = len(y)
    J_histoty = np.zeros((iters_n, 1))

    for i in range(iters_n):
        theta = theta - (alpha / m) * (x.T @ ((x @ theta) - y))
        J_histoty[i] = (x @ theta - y).T @ (x @ theta - y) / (2 * m)
    return theta, J_histoty


def normal_equation(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


def plot_J(J_history, iters_n):
    x = np.arange(1, iters_n + 1)
    plt.plot(x, J_history)
    plt.xlabel("iter numbers")
    plt.ylabel("cost value")
    plt.title("J function")
    plt.show()


# a test example
def predict(mu, sigma, theta, t):
    predict = np.array(t)
    final_predict = np.hstack((1, (predict - mu) / sigma))
    return final_predict @ theta


def main():
    alpha = 0.01
    iters_n = 1000
    x, x1, y, theta, mu, sigma = load_data()
    # plt.scatter(x[:, 1], x[:, 2])
    # plt.show()
    # theta, J_history = gradient_descent(x, y, theta, alpha, iters_n)
    # plot_J(J_history, iters_n)
    # print("mu:", mu)
    # print("sigma:", sigma)
    # print("theta:", theta.T)
    # print(predict(mu, sigma, theta))
    # theta = normal_equation(x1, y)
    # print("theta:", theta.T)
    # print(np.array((1, 1900, 4)) @ theta)

    for i in range(50):
        t = (1, np.random.randint(1500, 3000), np.random.randint(1, 5))
        theta, J_history = gradient_descent(x, y, theta, alpha, iters_n)
        p1 = predict(mu, sigma, theta, t[1:])
        theta = normal_equation(x1, y)
        p2 = np.array(t) @ theta
        if abs(p1 - p2) / (p1 + p2) < 0.001:
            print('ok')
        else:
            print('not good')


if __name__ == "__main__":
    main()
