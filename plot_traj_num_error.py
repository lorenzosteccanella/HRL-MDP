import pickle
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter

filehandler = open("list_errors", 'rb')
list_errors = pickle.load(filehandler)

# filehandler = open("list_errors3", 'rb')
# list_errors3 = pickle.load(filehandler)
#
# list_errors = list_errors2 + list_errors3
list_errors = sorted(list_errors, key=itemgetter(5))

x = []
y = []
for i, error in enumerate(list_errors):
    avg_error = np.mean(error[2])
    x.append(avg_error)
    y.append(list_errors[i][5])

plt.ylabel("absolute error")
plt.xlabel("n_of_trajectories")
plt.plot(y, x, "o-")
plt.show()