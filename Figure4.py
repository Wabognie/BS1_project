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
t = 5000
eta = 10

D = 80

beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))
tau = 0.05 ##a voir

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
Réalisation de la simulation pour Q = 0, les cellules sont totalement en désynchro.
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


fourierQ1list = []
fourierQ2list = []

for cell in range(nCell):
    #Centralisation et normalisation de b
    b1Q1norm = (bQ1[cell, :] - (max(bQ1[cell, :])/2))/(max(bQ1[cell, :])/2)

    b1Q2norm = (bQ2[cell, :] - (max(bQ2[cell, :])/2))/(max(bQ2[cell, :])/2)

    #Autocorrelation des valeurs : calcul la somme des multiplications des valeurs aux temps avec un décallage (b(t) * b(t+i))
    autocorrelQ1 = np.correlate(b1Q1norm, b1Q1norm, mode="full")
    autocorrelQ1 = autocorrelQ1[len(autocorrelQ1)//2:]

    autocorrelQ2 = np.correlate(b1Q2norm, b1Q2norm, mode="full")
    autocorrelQ2 = autocorrelQ2[len(autocorrelQ2)//2:]

    #soustraction de la moyenne des valeurs à toutes les valeurs :
    averageautocorr = sum(autocorrelQ1)/len(autocorrelQ1)
    autocorrelQ1 = autocorrelQ1 - averageautocorr

    averageautocorr = sum(autocorrelQ2)/len(autocorrelQ2)
    autocorrelQ2 = autocorrelQ2 - averageautocorr

    #Réalisation des transformées de fourier et récupération dans une liste
    fourierQ1list.append(np.fft.fft(autocorrelQ1))
    fourierQ2list.append(np.fft.fft(autocorrelQ2))

#calcul des fréquences fft
freqfour=np.fft.fftfreq(len(autocorrelQ1),d=tau)

bar.finish()
plt.show()

plt.plot(b1Q1norm)
plt.plot(b1Q2norm)
plt.title("Plot de b pour une bactérie")
plt.show()

plt.plot(autocorrelQ1)
plt.plot(autocorrelQ2)
plt.title("Autocorrelation")
plt.show()


#calcul de la moyenne des fft:
fourierQ1 = np.array([np.mean(k) for k in zip(*fourierQ1list)])
fourierQ2 = np.array([np.mean(k) for k in zip(*fourierQ2list)])


#Plot de la transfo de fourier
plt.plot(abs(freqfour), abs(fourierQ1), label = "Q = 1")
plt.plot(abs(freqfour), abs(fourierQ2), label = "Q = 0")
plt.xlabel('Frequence')
plt.ylabel('PSD')
plt.title("Fourier Transform Mean")
plt.legend()
plt.xlim(0,0.2)
plt.show()

plt.plot(abs(freqfour), abs(fourierQ1list[1]), label = "Q = 1")
plt.plot(abs(freqfour), abs(fourierQ2list[1]), label = "Q = 0")
plt.xlabel('Frequence')
plt.ylabel('PSD')
plt.title("Fourier Transform for 1 cell")
plt.legend()
plt.xlim(0,0.2)
plt.show()
