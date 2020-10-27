"""
BS1 project

retry all figures of article
"""

import numpy as np
import matplotlib.pyplot as plt
import random

n = 2

alpha = 216
kappa = 20
k_s0 = 1
k_s1 = 0.01


t = 10
eta = 2.0
Q = 0.4


a = np.zeros(t)
b = np.zeros(t)
c = np.zeros(t)

A = np.zeros(t)
B = np.zeros(t)
C = np.zeros(t)

S = np.zeros(t)
Se = np.zeros(t)


a[0] = 100
b[0] = 200
c[0] = 150

A[0] = 0
B[0] = 0
C[0] = 0

S[0] = 1
Se[0] = 1


for i in range(0,t-1):
    beta = random.gauss(1,0.05)

    a[i+1] = a[i] + t *(-a[i]+(alpha/(1+C[i]**n)))
    b[i+1] = b[i] + t *(-b[i]+(alpha/(1+A[i]**n)))
    c[i+1] = c[i] + t *(-c[i]+(alpha/(1+B[i]**n))+(kappa*S[i]/1+S[i]))

    A[i+1] = A[i]+t*(beta*(a[i]-A[i]))
    B[i+1] = B[i]+t*(beta*(b[i]-B[i]))
    C[i+1] = C[i]+t*(beta*(c[i]-C[i]))

    S[i+1] = S[i]+t*((-k_s0*S[i])+(k_s1*A[i])-(eta*(S[i]-Se[i])))

    Se[i+1] = Q*np.mean(S[0:i+1])

plt.plot(Se)
plt.show()
