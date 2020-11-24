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

D = 90

beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))
tau = 0.04 ##a voir

'''
Réalisation de la simulation pour Q = 1, les cellules sont en synchro parfaite.
'''
Q = 1

bar = Bar("Progress", max=t-1)
##initialisation des matrices
a = np.zeros((nCell, t))
bQ1 = np.zeros((nCell, t))
c = np.zeros((nCell, t))

A = np.zeros((nCell, t))
B = np.zeros((nCell, t))
C = np.zeros((nCell, t))

S = np.zeros((nCell, t))
Se = np.zeros((nCell, t))

a[:,0] = 0
bQ1[:,0] = 50
c[:,0] = 0

A[:,0] = 0
B[:,0] = 0
C[:,0] = 0

S[:,0] = 0
Se[:,0] = 0


time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        xi = random.gauss(0, D)
        betabis = beta[i] + (xi*tau)

        Se[i, j] = Q * np.mean(S[:, j])

        abis = a[i, j] + tau / 2 * (-a[i, j] + (alpha / (1 + C[i, j] ** n)))
        bbis = bQ1[i, j] + tau / 2 * (-bQ1[i, j] + (alpha / (1 + A[i, j] ** n)))
        cbis = c[i, j] + tau / 2 * (-c[i, j] + (alpha / (1 + B[i, j] ** n)) + (kappa * S[i, j] / 1 + S[i, j]))

        Abis = A[i, j] + tau / 2 * (betabis * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau / 2 * (betabis * (bQ1[i, j] - B[i, j]))
        Cbis = C[i, j] + tau / 2 * (betabis * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau / 2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i, j + 1] = a[i, j] + tau * (-abis + (alpha / (1 + Cbis ** n)))
        bQ1[i, j + 1] = bQ1[i, j] + tau * (-bbis + (alpha / (1 + Abis ** n)))
        c[i, j + 1] = c[i, j] + tau * (-cbis + (alpha / (1 + Bbis ** n)) + (kappa * Sbis / 1 + Sbis))

        A[i, j + 1] = A[i, j] + tau * (betabis * (abis - Abis))
        B[i, j + 1] = B[i, j] + tau * (betabis * (bbis - Bbis))
        C[i, j + 1] = C[i, j] + tau * (betabis * (cbis - Cbis))

        S[i, j + 1] = S[i, j] + tau * ((-k_s0 * Sbis) + (k_s1 * Abis) - (eta * (Sbis - Se[i, j])))

    bar.next()

'''
Réalisation de la simulation pour Q = 0, les cellules sont en synchro parfaite.
'''
Q = 0

bar = Bar("Progress", max=t-1)
##initialisation des matrices
a = np.zeros((nCell, t))
bQ2 = np.zeros((nCell, t))
c = np.zeros((nCell, t))

A = np.zeros((nCell, t))
B = np.zeros((nCell, t))
C = np.zeros((nCell, t))

S = np.zeros((nCell, t))
Se = np.zeros((nCell, t))

a[:,0] = 0
bQ2[:,0] = 50
c[:,0] = 0

A[:,0] = 0
B[:,0] = 0
C[:,0] = 0

S[:,0] = 0
Se[:,0] = 0


time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        xi = random.gauss(0, D)
        betabis = beta[i] + (xi*tau)

        Se[i, j] = Q * np.mean(S[:, j])

        abis = a[i, j] + tau / 2 * (-a[i, j] + (alpha / (1 + C[i, j] ** n)))
        bbis = bQ2[i, j] + tau / 2 * (-bQ2[i, j] + (alpha / (1 + A[i, j] ** n)))
        cbis = c[i, j] + tau / 2 * (-c[i, j] + (alpha / (1 + B[i, j] ** n)) + (kappa * S[i, j] / 1 + S[i, j]))

        Abis = A[i, j] + tau / 2 * (betabis * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau / 2 * (betabis * (bQ2[i, j] - B[i, j]))
        Cbis = C[i, j] + tau / 2 * (betabis * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau / 2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i, j + 1] = a[i, j] + tau * (-abis + (alpha / (1 + Cbis ** n)))
        bQ2[i, j + 1] = bQ2[i, j] + tau * (-bbis + (alpha / (1 + Abis ** n)))
        c[i, j + 1] = c[i, j] + tau * (-cbis + (alpha / (1 + Bbis ** n)) + (kappa * Sbis / 1 + Sbis))

        A[i, j + 1] = A[i, j] + tau * (betabis * (abis - Abis))
        B[i, j + 1] = B[i, j] + tau * (betabis * (bbis - Bbis))
        C[i, j + 1] = C[i, j] + tau * (betabis * (cbis - Cbis))

        S[i, j + 1] = S[i, j] + tau * ((-k_s0 * Sbis) + (k_s1 * Abis) - (eta * (Sbis - Se[i, j])))

    bar.next()


btotQ1 = []
for i in range(0, t):
    binter = 0
    for j in range(nCell):
        binter += bQ1[j,i]
    btotQ1.append(binter/nCell)

btotQ2 = []
for i in range(0, t):
    binter = 0
    for j in range(nCell):
        binter += bQ2[j,i]
    btotQ2.append(binter/nCell)

#Centralisation de btot et b1
btotQ1norm = (btotQ1 - (max(btotQ1)/2))/(max(btotQ1)/2)
b1Q1norm = (bQ1[1, :] - (max(bQ1[1, :])/2))/(max(bQ1[1, :])/2)

btotQ2norm = (btotQ2 - (max(btotQ2)/2))/(max(btotQ2)/2)
b1Q2norm = (bQ2[1, :] - (max(bQ2[1, :])/2))/(max(bQ2[1, :])/2)


#Autocorrelation des valeurs : calcul la somme des multiplications des valeurs aux temps avec un décallage (b(t) * b(t+i))
testAutocorrelQ1 = np.correlate(b1Q1norm, b1Q1norm, mode="full")
testAutocorrelQ1 = testAutocorrelQ1[len(testAutocorrelQ1)//2:]

testAutocorrelQ2 = np.correlate(b1Q2norm, b1Q2norm, mode="full")
testAutocorrelQ2 = testAutocorrelQ2[len(testAutocorrelQ2)//2:]

#soustraction de la moyenne des valeurs à toutes les valeurs :
averageautocorr = sum(testAutocorrelQ1)/len(testAutocorrelQ1)
testAutocorrelQ1 = testAutocorrelQ1 - averageautocorr

averageautocorr = sum(testAutocorrelQ2)/len(testAutocorrelQ2)
testAutocorrelQ2 = testAutocorrelQ2 - averageautocorr


#Autre test de plot direct d'une autocorrélation normalisée
#plot_acf(btotQ1, lags=np.arange(len(btot)))


#print(len(PSDlist))
#print((len(tot_freq)))

bar.finish()
plt.show()

plt.plot(b1Q1norm)
plt.plot(b1Q2norm)
plt.title("Plot de b pour une bactérie")
plt.show()

plt.plot(testAutocorrelQ1)
plt.plot(testAutocorrelQ2)
plt.title("Autocorrelation")
plt.show()

#print(len(tot_freq))

#plt.plot(PSDnorm)
#plt.show()
#calcul de la transformée de fourier sur l'autocorrelation
fourierQ1=np.fft.fft(testAutocorrelQ1)
fourierQ2=np.fft.fft(testAutocorrelQ2)


freqfour=np.fft.fftfreq(len(testAutocorrelQ1),d=tau)
#print(fourier)
#print(freqfour)

#Plot de la transfo de fourier
plt.plot(abs(freqfour), abs(fourierQ1), label = "Q = 1")
plt.plot(abs(freqfour), abs(fourierQ2), label = "Q = 0")
plt.xlabel('Frequence')
plt.ylabel('PSD')
plt.title("Fourier Transform")
plt.legend()
plt.xlim(0,0.2)

plt.show()
