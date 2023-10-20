# Dependencies
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv
data_2 = np.genfromtxt('ml_tree_out_800_2.csv', delimiter=',')[1:]
data_4 = np.genfromtxt('ml_tree_out_800_4.csv', delimiter=',')[1:]

# Plot data as line plot
plt.plot(data_2[:, 0], data_2[:, 1], 'o-', label='2 nodes')
plt.plot(data_4[:, 0], data_4[:, 1], 'o-', label='4 nodes')
plt.xlabel('Multiplication Factor')
plt.ylabel('Time (s)')
plt.title('Decision Tree Time vs Data Multiplication Factor')
plt.show()