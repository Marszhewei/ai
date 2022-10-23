#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))


def sigmoid_gradient(z):
    return (sigmoid(z) * (1 - sigmoid(z)))


def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y):
    length = nn_params.shape[0]
    theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape(
        hidden_layer_size, input_layer_size + 1
    )
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : length].reshape(
        num_labels, hidden_layer_size + 1
    )

    m = x.shape[0]
    class_y = np.zeros((m, num_labels))
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)

    a1 = np.hstack((np.ones((m, 1)), x))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    # cost function
    J = (-(((class_y.reshape(-1, 1)).T @ np.log(h.reshape(-1, 1)))
            + ((1 - class_y.reshape(-1, 1)).T @ np.log(1 - h.reshape(-1, 1)))) / m)
    return np.ravel(J)


def bp_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y):
    length = nn_params.shape[0]
    theta1 = (
        nn_params[0 : hidden_layer_size * (input_layer_size + 1)]
        .reshape(hidden_layer_size, input_layer_size + 1).copy())
    theta2 = (
        nn_params[hidden_layer_size * (input_layer_size + 1) : length]
        .reshape(num_labels, hidden_layer_size + 1).copy())
    m = x.shape[0]
    class_y = np.zeros((m, num_labels))
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)

    Theta2_x = theta2[:, 1:theta2.shape[1]]
    theta1_grad = np.zeros((theta1.shape))
    theta2_grand = np.zeros((theta2.shape))

    a1 = np.hstack((np.ones((m, 1)), x))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    delta3 = np.zeros((m, num_labels))
    delta2 = np.zeros((m, hidden_layer_size))
    for i in range(m):
        delta3[i, :] = h[i, :] - class_y[i, :]
        theta2_grand = theta2_grand + (delta3[i, :].reshape(1, -1).T @ a2[i, :].reshape(1, -1))
        delta2[i, :] = (delta3[i, :].reshape(1, -1) @ Theta2_x) * sigmoid_gradient(z2[i, :])
        theta1_grad = theta1_grad + ((delta2[i, :].reshape(1, -1)).T @ a1[i, :].reshape(1, -1))

    return np.ravel((np.vstack((theta1_grad.reshape(-1, 1), theta2_grand.reshape(-1, 1)))) / m)


def init_weight(fan_in, fan_out):
    w = np.zeros((fan_out, fan_in + 1))
    x = np.arange(1, fan_out * (fan_in + 1) + 1)
    return np.sin(x).reshape(w.shape) / 10


def gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y):
    grad = np.zeros((nn_params.shape[0]))
    step = np.zeros((nn_params.shape[0]))
    e = 1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss1 = cost_function(
            nn_params - step.reshape(-1, 1), input_layer_size, hidden_layer_size, num_labels, x, y)
        loss2 = cost_function(
            nn_params + step.reshape(-1, 1), input_layer_size, hidden_layer_size, num_labels, x, y)
        grad[i] = (loss2 - loss1) / (2 * e)
        step[i] = 0
    return grad


def main():
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    theta1 = init_weight(input_layer_size, hidden_layer_size)
    theta2 = init_weight(hidden_layer_size, num_labels)
    x = init_weight(input_layer_size - 1, m)
    y = (np.mod(np.arange(1, m + 1), num_labels)).T

    y = y.reshape(-1, 1)
    nn_params = np.vstack((theta1.reshape(-1, 1), theta2.reshape(-1, 1)))
    # use BP
    grad = bp_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y)
    # direct calculation
    num_grad = gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y)
    print(np.hstack((num_grad.reshape(-1, 1), grad.reshape(-1, 1))))


if __name__ == "__main__":
    main()
