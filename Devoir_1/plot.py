# Dependencies
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv
# data_2 = np.genfromtxt('out/local_node_comp.csv', delimiter=',')[1:]
data_4 = np.genfromtxt('out/ml_tree_600_4_8.csv', delimiter=',')[1:]

# for i in [1, 2, 4, 6, 8, 10]:
#     f = f'out/ml_tree_600_{i}_8.csv'
#     data = np.genfromtxt(f, delimiter=',')[1:]
#     sca_data = data[:, 1] / data[:, 1][0]
#     plt.plot(data[:, 0], sca_data, 'o-', label=f'{i} nodes')

# x_nodes = [1, 2, 4, 6, 8, 10]
# y_nodes = []
# for i in x_nodes:
#     f = f'out/ml_tree_600_{i}_8.csv'
#     data = np.genfromtxt(f, delimiter=',')[1:]
#     y_nodes.append(data[:, 1][-1])
# plt.plot(x_nodes, y_nodes, 'o-')

# Plot data as line plot
# plt.plot(data_2[:, 0]*1272824/1000000, data_2[:, 0], 'o-')
# plt.plot(data_2[:, 0]*1272824/1000000, data_2[:, 1]/data_2[:, 1][0], 'o-')
plt.plot(data_4[:, 0]*1272824/1000000, data_4[:, 2], 'o-')
# plt.plot(data_2[:, 0], data_2[:, 1], 'o-')
# plt.plot(data_4[:, 0], data_4[:, 1], 'o-', label='4 nodes')
# plt.xlabel('DataSet size (Mb)')
# plt.ylabel('Scale Factor')
# plt.title('Scale Factor vs DataSet size on 4 nodes')
# plt.xlabel('Nombre de noeuds')
# plt.ylabel("Temp d'exécution (s)")
# plt.title('Nombre de noeuds vs Temps d\'exécution sur 827 Mb de données')
plt.ylabel("Accuracy")
plt.title('Accuracy dun modèle de classification en fonction du volume de données')
plt.show()
#827.3356