
import math
import numpy as np
'''
def euc_dist(a,b):
    return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(p[0],q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), euc_dist(p[i],q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), euc_dist(p[0],q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i-1, j, p, q),_c(ca, i-1, j-1, p, q),_c(ca, i, j-1, p, q)),euc_dist(p[i],q[j]))
    else:
        ca[i, j] = float('inf')

    return ca[i, j]

def frechet_distance(p,q):
    ca = np.ones((len(p),len(q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(p)-1, len(q)-1, p, q)

#curve1 = [[6,6],[0,6],[0,0],[6,0],[6,6]]
curve1 = [[6,6],[4,6],[2,6],[0,6],[0,4],[0,2],[0,0],[2,0],[4,0],[6,0],[6,2],[6,4],[6,6]]
#curve2 = [[4,4],[1,4],[1,1],[4,1],[4,4]]
curve2 = [[4,4],[3,4],[2,4],[1,4],[1,3],[1,2],[1,1],[2,1],[3,1],[4,1],[4,2],[4,3],[4,4]]
result = frechet_distance(curve1, curve2)
print(result)


import sys
sys.setrecursionlimit(2000)
print(sys.getrecursionlimit())


import numpy as np
import similaritymeasures
import matplotlib.pyplot as plt

#curve1 = [[6,6],[0,6],[0,0],[6,0],[6,6]]
#curve1 = [[6,6],[4,6],[2,6],[0,6],[0,4],[0,2],[0,0],[2,0],[4,0],[6,0],[6,2],[6,4],[6,6]]
#curve2 = [[4,4],[1,4],[1,1],[4,1],[4,4]]
#curve2 = [[4,4],[3,4],[2,4],[1,4],[1,3],[1,2],[1,1],[2,1],[3,1],[4,1],[4,2],[4,3],[4,4]]

# Generate random experimental data

exp_data = np.zeros((len(curve1), 2))
for i in range(len(curve1)):
    exp_data[i] = curve1[i]

# Generate random numerical data

num_data = np.zeros((len(curve2), 2))
for j in range(len(curve2)):
    num_data[j] = curve2[j]

# quantify the difference between the two curves using PCM
pcm = similaritymeasures.pcm(exp_data, num_data)

# quantify the difference between the two curves using
# Discrete Frechet distance
df = similaritymeasures.frechet_dist(exp_data, num_data)

# quantify the difference between the two curves using
# area between two curves
area = similaritymeasures.area_between_two_curves(exp_data, num_data)

# quantify the difference between the two curves using
# Curve Length based similarity measure
cl = similaritymeasures.curve_length_measure(exp_data, num_data)

# quantify the difference between the two curves using
# Dynamic Time Warping distance
dtw, d = similaritymeasures.dtw(exp_data, num_data)

# print the results
print(pcm, df, area, cl, dtw)

# plot the data
plt.figure()
plt.plot(exp_data[:, 0], exp_data[:, 1])
plt.plot(num_data[:, 0], num_data[:, 1])
plt.show()
'''

a = [10,10,5,5]
a = [a[0] / a[3],a[1] / a[3],a[2] / a[3],a[3] / a[3]]
print(a)