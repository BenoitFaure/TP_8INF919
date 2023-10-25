# Dependencies
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv
data_2 = np.genfromtxt('out/ml_tree_local_bck.csv', delimiter=',')[1:]
# data_4 = np.genfromtxt('ml_tree_out_800_4.csv', delimiter=',')[1:]

# for i in [1, 2, 4, 6, 8, 10]:
#     f = f'out/ml_tree_600_{i}_8.csv'
#     data = np.genfromtxt(f, delimiter=',')[1:]
#     # sca_data = data[:, 1] / data[:, 1][0]
#     plt.plot(data[:, 0], data[:, 1], 'o-', label=f'{i} nodes')

# Plot data as line plot
plt.plot(data_2[:, 0], data_2[:, 1], 'o-', label='2 nodes')
# plt.plot(data_4[:, 0], data_4[:, 1], 'o-', label='4 nodes')
plt.xlabel('Multiplication Factor')
plt.ylabel('Time (s)')
plt.title('Decision Tree Time vs Data Multiplication Factor')
plt.show()