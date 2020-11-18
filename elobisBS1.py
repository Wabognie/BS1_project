import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.signal import find_peaks
from progress.bar import Bar

nCell = 10000
##constantes
n = 2
alpha = 112
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 600
eta = 2.0
Q = 0.8

beta = []
for i in range(nCell):
    beta.append(random.gauss(1,0.05))


tau = 0.33 ##a voir

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

    bar.next()

tot_freq=[]
for i in range(nCell):
    #plt.plot(time,b[i])

    peaks, _ = find_peaks(b[i])
    peaks_l, _ = find_peaks(-b[i])
    periods = []
    freqs = []
    n = 0
    for x in peaks :
        if n < len(peaks)-1 and n < len(peaks_l)-1:
            period = peaks[n+1]-peaks[n]
            period_l = peaks_l[n+1]-peaks_l[n]
            freqs.append(period)
            freqs.append(period_l)
        else :
            break
        n+=1
    tot_freq.append(1/np.mean(freqs))


fig, axs = plt.subplots(2)
fig.subplots_adjust(hspace= 0.7)
#fig.suptitle("alph a: "+str(alpha)+", b : "+str(b[0,0])+", q :"+str(Q))


cellplot = [random.randint(0,nCell-1) for p in range(0,10)]
for i in cellplot :
    #plt.plot(time, b[i])
    axs[0].plot(time, b[i])
    axs[0].set_title('Variation in cI production' + '\n'+'for 10 among 10,000 cells with cell density of ' + str(Q))
    axs[0].set_ylabel('cI mRNA level' +'\n'+'(arbitrary units)')
    axs[0].set_xlabel('time(min)')


#plt.savefig("./photos/euler_implicite_alpha_"+str(alpha)+"_b_"+str(b[0,0])+"_q_"+str(Q)+".jpg")
bar.finish()
#plt.show()
print(len(tot_freq))

#plt.hist(tot_freq, bins = 50, range =(0.02,0.03))
axs[1].hist(tot_freq, bins = 20, range =(0.022,0.026))
axs[1].set_title('Oscillation frequencies of cI production' + '\n' + 'for 10,000 cells with cell density of ' + str(Q))
axs[1].set_ylabel('number of cells')
axs[1].set_xlabel('frequency(min-1)')
plt.savefig("./photosreun/euler_rangekuta_alpha_"+str(alpha)+"_b_"+str(b[0,0])+"_q_"+str(Q)+".jpg")
