# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rv_continuous



N=3
N=3
A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
Z = A + 1j * B
print(Z)
# Step 2
Q, R = np.linalg.qr(Z)
print(Q)
print(R)
# Step 3
Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

New_matrix=np.dot(Q, Lambda)
print(New_matrix)