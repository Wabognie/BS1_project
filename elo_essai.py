import numpy as np
import matplotlib.pyplot as plt
import random

nCell = 10
##constantes
n = 2
alpha = 216
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 600
eta = 2.0
Q = 0.8
beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))
tau = 0.5 ##a voir

##initialisation des matrices
a = np.zeros((nCell, t))
b = np.zeros((nCell, t))
c = np.zeros((nCell, t))

A = np.zeros((nCell, t))
B = np.zeros((nCell, t))
C = np.zeros((nCell, t))

S = np.zeros((nCell, t))
Se = np.zeros((nCell, t))

a[:,0] = 0
b[:,0] = 0
c[:,0] = 0

A[:,0] = 0
B[:,0] = 0
C[:,0] = 0

S[:,0] = 0
Se[:,0] = 0

time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        a[i,j+1] = a[i,j] + tau *(-a[i,j]+(alpha/(1+C[i,j]**n)))
        b[i,j+1] = b[i,j] + tau *(-b[i,j]+(alpha/(1+A[i,j]**n)))
        c[i,j+1] = c[i,j] + tau *(-c[i,j]+(alpha/(1+B[i,j]**n))+(kappa*S[i,j]/1+S[i,j]))

        A[i,j+1] = A[i,j]+ tau*(beta[i]*(a[i,j]-A[i,j]))
        B[i,j+1] = B[i,j]+ tau*(beta[i]*(b[i,j]-B[i,j]))
        C[i,j+1] = C[i,j]+ tau*(beta[i]*(c[i,j]-C[i,j]))

        S[i,j+1] = S[i,j]+tau*((-k_s0*S[i,j])+(k_s1*A[i,j])-(eta*(S[i,j]-Se[i,j])))

        Se[i,j] = Q*np.mean(S[:,j])

#plt.plot(time,amplitude_a,label="a[i]")
for i in range(nCell):
    plt.plot(time,b[i])
#plt.plot(time,amplitude_c, label="c[i]")
#plt.plot(time, amplitude_Se, label="Se[i]")
#plt.plot(time,amplitude_S, label ="S[i]")

plt.show()
