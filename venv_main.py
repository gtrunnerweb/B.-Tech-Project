import matplotlib.pyplot as ploter
import numpy as npy
import pandas as pds

read_data = pds.read_csv('Google.csv')
observed_prices = read_data['Close'].values

Sample_time = 1
num_of_obv = len(observed_prices)
Q = npy.array([[0.01, 0], [0, 0.01]])
R = 0.185
Φ = npy.array([[1, Sample_time], [0, 1]])
Γ = npy.array([[0.5 * Sample_time ** 2], [Sample_time]]).T
H = npy.array([[1, 0]])
Est_P = npy.eye(2)
white_noise = npy.random.randn(num_of_obv)
X = npy.zeros((2, num_of_obv))
Est_X = npy.zeros((2, num_of_obv))


for time_step in range(1, num_of_obv):
    Est_P = npy.dot(npy.dot(Φ, Est_P), Φ.T) + Q
    Est_X[:, time_step] = npy.dot(Φ, Est_X[:, time_step - 1]) + npy.dot(Γ, white_noise[time_step])

    Kalman_Gain = npy.dot(npy.dot(Est_P, H.T), npy.linalg.inv(npy.dot(npy.dot(H, Est_P), H.T) + R))

    Est_P = npy.dot((npy.eye(2) - npy.dot(Kalman_Gain, H)), Est_P)
    epsilon_residual = observed_prices[time_step] - npy.dot(H, Est_X[:, time_step])
    Est_X[:, time_step] = Est_X[:, time_step] + npy.dot(Kalman_Gain, epsilon_residual)


ploter.figure(figsize=(15, 7))
ploter.xlabel('Time Step of Each Closing Price')
ploter.ylabel('Stock Price')
ploter.plot(observed_prices, label='Actual Historical Stock Prices', color='blue', alpha=0.7, marker='o', markersize=4)
ploter.plot(Est_X[0], label='Kalman Filter Estimate', color='red', alpha=0.7, marker='o', markersize=4)

ploter.title('Kalman Filter Estimate vs Actual Historical Stock Prices')


for iter in range(num_of_obv):
    ploter.annotate(f'{Est_X[0, iter]:.2f}', (iter, Est_X[0, iter]), textcoords="offset points", xytext=(-10,11), ha='center', fontsize=7)
    ploter.annotate(f'{observed_prices[iter]:.2f}', (iter, observed_prices[iter]), textcoords="offset points", xytext=(-10,-25), ha='center', fontsize=7)


ploter.legend()
ploter.grid(True)
ploter.show()