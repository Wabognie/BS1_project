import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.signal import find_peaks
from progress.bar import Bar


nCell = 10000
##constantes
n = 2
alpha = 216
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 600
eta = 2.0
Q = 0.4

beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))
tau = 0.4 ##a voir

bar = Bar("Progress", max=t-1)
##initialisation des matrices
a = np.zeros((nCell, t))
b = np.zeros((nCell, t))
c = np.zeros((nCell, t))

A = np.zeros((nCell, t))
B = np.zeros((nCell, t))
C = np.zeros((nCell, t))

S = np.zeros((nCell, t))
Se = np.zeros((nCell, t))


a[:,0] = random.randint(1,300)
b[:,0] = random.randint(1,300)
c[:,0] = random.randint(1,300)

A[:,0] = 0
B[:,0] = 0
C[:,0] = 0

S[:,0] = 0
Se[:,0] = 0

"""
a[:,0] = random.randint(1,100)
b[:,0] = random.randint(1,100)
c[:,0] = random.randint(1,100)

A[:,0] = random.randint(1,100)
B[:,0] = random.randint(1,100)
C[:,0] = random.randint(1,100)

S[:,0] = random.randint(1,100)
Se[:,0] = 0
"""
"""
print(a[:,0])
print(b[:,0])
print(c[:,0])
print(A[:,0])
print(B[:,0])
print(C[:,0])
"""

"""
a[:,0] = 0
b[:,0] = 0
c[:,0] = 0

A[:,0] = 0
B[:,0] = 0
C[:,0] = 0

S[:,0] = 0
Se[:,0] = 0
"""

frequence_list = []

time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        Se[i, j] = Q * np.mean(S[:, j])

        a[i,j+1] = a[i,j] + tau *(-a[i,j]+(alpha/(1+C[i,j]**n)))
        b[i,j+1] = b[i,j] + tau *(-b[i,j]+(alpha/(1+A[i,j]**n)))
        c[i,j+1] = c[i,j] + tau *(-c[i,j]+(alpha/(1+B[i,j]**n))+(kappa*S[i,j]/1+S[i,j]))

        A[i,j+1] = A[i,j] + tau*(beta[i]*(a[i,j]-A[i,j]))
        B[i,j+1] = B[i,j] + tau*(beta[i]*(b[i,j]-B[i,j]))
        C[i,j+1] = C[i,j] + tau*(beta[i]*(c[i,j]-C[i,j]))

        S[i,j+1] = S[i,j] + tau*((-k_s0*S[i,j])+(k_s1*A[i,j])-(eta*(S[i,j]-Se[i,j])))

        a[i,j] = a[i,j+1] - tau *(-a[i,j+1]+(alpha/(1+C[i,j+1]**n)))
        b[i,j] = b[i,j+1] - tau *(-b[i,j+1]+(alpha/(1+A[i,j+1]**n)))
        c[i,j] = c[i,j+1] - tau *(-c[i,j+1]+(alpha/(1+B[i,j+1]**n))+(kappa*S[i,j+1]/1+S[i,j+1]))

        A[i,j] = A[i,j+1] - tau*(beta[i]*(a[i,j+1]-A[i,j+1]))
        B[i,j] = B[i,j+1] - tau*(beta[i]*(b[i,j+1]-B[i,j+1]))
        C[i,j] = C[i,j+1] - tau*(beta[i]*(c[i,j+1]-C[i,j+1]))

        S[i,j] = S[i,j+1] - tau*((-k_s0*S[i,j+1])+(k_s1*A[i,j+1])-(eta*(S[i,j+1]-Se[i,j+1])))

    bar.next()

freqs = []
for i in range(nCell):
    plt.plot(time,b[i])

    peaks, _ = find_peaks(b[i])
    periods = []

    n = 0
    for x in peaks :
        if n < len(peaks)-1 :
            period = peaks[n+1]-peaks[n]
            periods.append(period)
            freq = 1/period
            freq = round(freq,5)
            freqs.append(freq)
        else :
            break
        n+=1


plt.show()
print(freqs)

bar.finish()
plt.hist(freqs, bins = 50, density = True)
plt.show()
