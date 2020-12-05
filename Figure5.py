###Figure 5
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
t = 2000
eta = 2



beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))

tau = 0.35 ##a voir
D = 0.4

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

for cell in range(nCell):
    bQ1[cell,0] = random.randint(50, 100)
    a[cell,0] = random.randint(50, 100)
    c[cell,0] = random.randint(50, 100)

    A[cell,0] = random.randint(50, 100)
    B[cell,0] = random.randint(50, 100)
    C[cell,0] = random.randint(50, 100)

    S[cell,0] = random.randint(50, 100)
    Se[cell,0] = random.randint(50, 100)


time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        xiA = random.gauss(0, D)
        betabisA = beta[i] + (xiA*tau)

        xiB = random.gauss(0, D)
        betabisB = beta[i] + (xiB*tau)

        xiC = random.gauss(0, D)
        betabisC = beta[i] + (xiC*tau)

        abis = a[i, j] + tau / 2 * (-a[i, j] + (alpha / (1 + C[i, j] ** n)))
        bbis = bQ1[i, j] + tau / 2 * (-bQ1[i, j] + (alpha / (1 + A[i, j] ** n)))
        cbis = c[i, j] + tau / 2 * (-c[i, j] + (alpha / (1 + B[i, j] ** n)) + (kappa * S[i, j] / 1 + S[i, j]))

        Abis = A[i, j] + tau / 2 * (betabisA * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau / 2 * (betabisB * (bQ1[i, j] - B[i, j]))
        Cbis = C[i, j] + tau / 2 * (betabisC * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau / 2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i, j + 1] = a[i, j] + tau * (-abis + (alpha / (1 + Cbis ** n)))
        bQ1[i, j + 1] = bQ1[i, j] + tau * (-bbis + (alpha / (1 + Abis ** n)))
        c[i, j + 1] = c[i, j] + tau * (-cbis + (alpha / (1 + Bbis ** n)) + (kappa * Sbis / 1 + Sbis))

        A[i, j + 1] = A[i, j] + tau * (betabisA * (abis - Abis))
        B[i, j + 1] = B[i, j] + tau * (betabisB * (bbis - Bbis))
        C[i, j + 1] = C[i, j] + tau * (betabisC * (cbis - Cbis))

        S[i, j + 1] = S[i, j] + tau * ((-k_s0 * Sbis) + (k_s1 * Abis) - (eta * (Sbis - Se[i, j])))

        Se[i, j+1] = Q * np.mean(S[:, j])
    bar.next()

'''
Réalisation de la simulation pour Q = 0, les cellules sont totalement en désynchro.
'''
Q = 0.2
beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))

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


for cell in range(nCell):
    bQ2[cell,0] = random.randint(50, 100)
    a[cell,0] = random.randint(50, 100)
    c[cell,0] = random.randint(50, 100)

    A[cell,0] = random.randint(50, 100)
    B[cell,0] = random.randint(50, 100)
    C[cell,0] = random.randint(50, 100)

    S[cell,0] = random.randint(50, 100)
    Se[cell,0] = random.randint(50, 100)


time = np.arange(0,t)

for j in range(0, t-1):
    for i in range(0, nCell):
        xiA = random.gauss(0, D)
        betabisA = beta[i] + (xiA*tau)

        xiB = random.gauss(0, D)
        betabisB = beta[i] + (xiB*tau)

        xiC = random.gauss(0, D)
        betabisC = beta[i] + (xiC*tau)

        abis = a[i, j] + tau / 2 * (-a[i, j] + (alpha / (1 + C[i, j] ** n)))
        bbis = bQ2[i, j] + tau / 2 * (-bQ2[i, j] + (alpha / (1 + A[i, j] ** n)))
        cbis = c[i, j] + tau / 2 * (-c[i, j] + (alpha / (1 + B[i, j] ** n)) + (kappa * S[i, j] / 1 + S[i, j]))

        Abis = A[i, j] + tau / 2 * (betabisA * (a[i, j] - A[i, j]))
        Bbis = B[i, j] + tau / 2 * (betabisB * (bQ2[i, j] - B[i, j]))
        Cbis = C[i, j] + tau / 2 * (betabisC * (c[i, j] - C[i, j]))

        Sbis = S[i, j] + tau / 2 * ((-k_s0 * S[i, j]) + (k_s1 * A[i, j]) - (eta * (S[i, j] - Se[i, j])))

        a[i, j + 1] = a[i, j] + tau * (-abis + (alpha / (1 + Cbis ** n)))
        bQ2[i, j + 1] = bQ2[i, j] + tau * (-bbis + (alpha / (1 + Abis ** n)))
        c[i, j + 1] = c[i, j] + tau * (-cbis + (alpha / (1 + Bbis ** n)) + (kappa * Sbis / 1 + Sbis))

        A[i, j + 1] = A[i, j] + tau * (betabisA * (abis - Abis))
        B[i, j + 1] = B[i, j] + tau * (betabisB * (bbis - Bbis))
        C[i, j + 1] = C[i, j] + tau * (betabisC * (cbis - Cbis))

        S[i, j + 1] = S[i, j] + tau * ((-k_s0 * Sbis) + (k_s1 * Abis) - (eta * (Sbis - Se[i, j])))

        Se[i, j+1] = Q * np.mean(S[:, j])
    bar.next()


fourierQ1list = []
fourierQ2list = []
autoQ1 = []
autoQ2 = []

max = ''
min =''
for cell in range(nCell):
    #Centralisation et normalisation de b
    b1Q1norm = (bQ1[cell, 0:2000] - (np.mean(bQ1[cell, 0:2000])))/(np.var(bQ1[cell, 0:2000]))
    b1Q2norm = (bQ2[cell, 0:2000] - (np.mean(bQ2[cell, 0:2000])))/(np.var(bQ2[cell, 0:2000]))

    autoCorrelQ1 = np.correlate(b1Q1norm, b1Q1norm, mode="full")
    autoCorrelQ1 = autoCorrelQ1[autoCorrelQ1.size//2:]

    autoCorrelQ2 = np.correlate(b1Q2norm, b1Q2norm, mode="full")
    autoCorrelQ2 = autoCorrelQ2[autoCorrelQ2.size//2:]

    autoQ1.append(autoCorrelQ1)
    autoQ2.append(autoCorrelQ2)

    #spectro.append(abs(autoCorrelQ1))
    #spectro2.append(abs(autoCorrelQ2))
    #Réalisation des transformées de fourier et récupération dans une liste
    fourierQ1list.append(np.real(np.fft.fft(autoCorrelQ1)))
    fourierQ2list.append(np.real(np.fft.fft(autoCorrelQ2)))




plt.imshow(np.transpose(fourierQ1list), extent = [0,nCell,0,2], aspect= 'auto',vmin=1, vmax=3, cmap='rainbow')
plt.yticks(np.arange(0,0.25,0.1))
plt.ylim(0.07,0.25)
plt.colorbar()
plt.ylabel('frequency (arb. unit)')
plt.xlabel('repressilators')
plt.show()

plt.imshow(np.transpose(fourierQ2list), extent = [0,nCell,0,2],aspect='auto',vmin=1, vmax=3, cmap='rainbow')
plt.yticks(np.arange(0,0.25,0.1))
plt.ylim(0.07,0.25)
plt.ylabel('frequency (arb. unit)')
plt.xlabel('repressilators')
plt.colorbar()
plt.show()
