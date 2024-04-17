import numpy as np
import matplotlib.pyplot as plt

from os import listdir

improvements = []

for fn in listdir("results/"):
    num_qubits, times_saving, times_without_saving = np.load(f"./results/{fn}")
    improvement = [times_without_saving[i]/times_saving[i] for i in range(len(times_saving))]
    improvements.append(improvement)
    plt.scatter(num_qubits, improvement, color='r', alpha=0.3, edgecolors='none')

average_improvement = np.array([0.0]*len(improvements[0]))
for imp in improvements:
    average_improvement += imp
average_improvement /= len(improvements)

plt.scatter(num_qubits, average_improvement, label="average improvement", color='r', alpha=1)
plt.xticks(num_qubits)
plt.xlabel("Number of qubits")
plt.ylabel("time_without_saving / time_saving")
plt.title("Improvement in recompilation times with vs without saving\nprevious layer MPS, for random circuits of 3 to 7 qubits")
plt.legend()
plt.show()
