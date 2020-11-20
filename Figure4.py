import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.signal import find_peaks
from progress.bar import Bar
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf

nCell = 100
##constantes
n = 2
alpha = 111
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 4500
eta = 2.0
Q = 0

D = 75

beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))
tau = 0.04 ##a voir

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


a[:,0] = 0
b[:,0] = 50
c[:,0] = 0

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
        xi = random.gauss(0, D)
        betabis = beta[i] + (xi*tau)

        Se[i, j] = Q * np.mean(S[:, j])

        abis = a[i, j] + tau / 2 * (-a[i, j] + (alpha / (1 + C[i, j] ** n)))
        bbis = b[i, j] + tau / 2 * (-b[i, j] + (alpha / (1 + A[i, j] ** n)))
        cbis = c[i, j] + tau / 2 * (-c[i, j] + (alpha / (1 + B[i, j] ** n)) + (kappa * S[i, j] / 1 + S[i, j]))

        Abis = A[i, j] + tau / 2 * (betabis * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau / 2 * (betabis * (b[i, j] - B[i, j]))
        Cbis = C[i, j] + tau / 2 * (betabis * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau / 2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i, j + 1] = a[i, j] + tau * (-abis + (alpha / (1 + Cbis ** n)))
        b[i, j + 1] = b[i, j] + tau * (-bbis + (alpha / (1 + Abis ** n)))
        c[i, j + 1] = c[i, j] + tau * (-cbis + (alpha / (1 + Bbis ** n)) + (kappa * Sbis / 1 + Sbis))

        A[i, j + 1] = A[i, j] + tau * (betabis * (abis - Abis))
        B[i, j + 1] = B[i, j] + tau * (betabis * (bbis - Bbis))
        C[i, j + 1] = C[i, j] + tau * (betabis * (cbis - Cbis))

        S[i, j + 1] = S[i, j] + tau * ((-k_s0 * Sbis) + (k_s1 * Abis) - (eta * (Sbis - Se[i, j])))

    bar.next()


btot = []
for i in range(0, t):
    binter = 0
    for j in range(nCell):
        binter += b[j,i]
    btot.append(binter/nCell)

theMaxOfbtot = 0
for num in btot:
    if num > theMaxOfbtot:
        theMaxOfbtot = num
#print(btot)
btotnorm = (btot - (theMaxOfbtot/2))/(theMaxOfbtot/2)
b1norm = (b[1, :] - (max(b[1, :])/2))/(max(b[1, :])/2)
#print(btot)
#print(len(btot))


#calcul du décallage de fréquence
PSDlist = []
PSDmax = 0
#PSDfreq = []
for i in range(0, t-1):
    PSD = 0
    for j in range(0, t-i):
            PSD += btotnorm[j] * btotnorm[j+i]
    PSDlist.append(PSD)
    if PSD > PSDmax:
        PSDmax = PSD

PSDnorm = PSDlist/PSDmax
    #if i == 0: PSDfreq.append(0)
    #else: PSDfreq.append((1/i*tau)*60)

testAutocorrel = np.correlate(b1norm,b1norm,mode="full")
testAutocorrel = testAutocorrel[len(testAutocorrel)//2:]
averageautocorr = sum(testAutocorrel)/len(testAutocorrel)

testAutocorrel = testAutocorrel - averageautocorr

plot_acf(btot, lags=np.arange(len(btot)))


#print(len(PSDlist))
#print((len(tot_freq)))

bar.finish()
plt.show()

plt.plot(b1norm)
plt.show()

plt.plot(testAutocorrel)
plt.show()

#print(len(tot_freq))

plt.plot(PSDnorm)
plt.show()

fourier=np.fft.fft(testAutocorrel)
freqfour=np.fft.fftfreq(len(testAutocorrel),d=tau)
print(fourier)
print(freqfour)
plt.xlabel('Frequence')
plt.ylabel('PSD')
plt.title("Fourier Transform")
plt.plot(abs(freqfour), abs(fourier))
plt.show()