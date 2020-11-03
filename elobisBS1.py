import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import numpy.fft as fft

nCell = 1000
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
tau = 0.4 ##a voir

##initialisation des matrices
a = np.zeros((nCell, t))
b = np.zeros((nCell, t))
c = np.zeros((nCell, t))

A = np.zeros((nCell, t))
B = np.zeros((nCell, t))
C = np.zeros((nCell, t))

S = np.zeros((nCell, t))
Se = np.zeros((nCell, t))

a[:,0] = random.randint(0,300)
b[:,0] = random.randint(0,300)
c[:,0] = random.randint(0,300)

A[:,0] = random.randint(0,300)
B[:,0] = random.randint(0,300)
C[:,0] = random.randint(0,300)

S[:,0] = random.randint(0,300)
Se[:,0] = random.randint(0,300)

frequence_list = []

time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        Se[i, j] = Q * np.mean(S[:, j])

        abis = a[i,j] + tau/2 *(-a[i,j]+(alpha/(1+C[i,j]**n)))
        bbis = b[i,j] + tau/2 *(-b[i,j]+(alpha/(1+A[i,j]**n)))
        cbis = c[i,j] + tau/2 *(-c[i,j]+(alpha/(1+B[i,j]**n))+(kappa*S[i,j]/1+S[i,j]))

        Abis = A[i, j] + tau/2 * (beta[i] * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau/2 * (beta[i] * (b[i, j] - B[i, j]))
        Cbis = C[i, j] + tau/2 * (beta[i] * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau/2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i,j+1] = a[i,j] + tau *(-abis+(alpha/(1+Cbis**n)))
        b[i,j+1] = b[i,j] + tau *(-bbis+(alpha/(1+Abis**n)))
        c[i,j+1] = c[i,j] + tau *(-cbis+(alpha/(1+Bbis**n))+(kappa*Sbis/1+Sbis))

        A[i,j+1] = A[i, j]+ tau*(beta[i]*(abis-Abis))
        B[i,j+1] = B[i, j]+ tau*(beta[i]*(bbis-Bbis))
        C[i,j+1] = C[i, j]+ tau*(beta[i]*(cbis-Cbis))

        S[i,j+1] = S[i,j]+tau*((-k_s0*Sbis)+(k_s1*Abis)-(eta*(Sbis-Se[i,j])))

#plt.plot(time,amplitude_a,label="a[i]")
for i in range(nCell):
    #plt.plot(time,b[i])

    T = 60/(t/len(b[i]))
    a = np.abs(fft.rfft(b[i], n=b[i].size))
    a[0]=0
    freqs = fft.rfftfreq(b[i].size, d=1./T)
    freqs = np.divide(60,freqs)
    max_freq = freqs[np.argmax(a)]
    p = round(1/max_freq, 3)

    frequence_list.append(p)
#plt.plot(time,amplitude_c, label="c[i]")
#plt.plot(time, amplitude_Se, label="Se[i]")
#plt.plot(time,amplitude_S, label ="S[i]")

#plt.show()
plt.hist(frequence_list)
plt.show()
