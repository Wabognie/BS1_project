"""
BS1 project

retry all figures of article
repressilator, transpose idea : http://be150.caltech.edu/2020/content/lessons/08_repressilator.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.fft as fft

##constantes
n = 2
alpha = 216
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 600
eta = 2.0
Q = 0.4

tau = 0.4

nb_cells = 10
Se = np.zeros(t)
Se[0] = 0

S_dict={}
p_dict={}
for cells in range(0,nb_cells):
    beta = random.gauss(1,0.05)
    ##initialisation des matrices
    a = np.zeros(t)
    b = np.zeros(t)
    c = np.zeros(t)

    A = np.zeros(t)
    B = np.zeros(t)
    C = np.zeros(t)

    S = np.zeros(t)


    a[0] = 0
    b[0] = 0
    c[0] = 0

    A[0] = 0
    B[0] = 0
    C[0] = 0

    S[0] = 0

    time = np.arange(0,t)

    for i in range(0,t-1):
        #print(S)
        #print("Se : " + str(Se))
        a[i+1] = a[i] + tau *(-a[i]+(alpha/(1+C[i]**n)))
        b[i+1] = b[i] + tau *(-b[i]+(alpha/(1+A[i]**n)))
        c[i+1] = c[i] + tau *(-c[i]+(alpha/(1+B[i]**n))+(kappa*S[i]/1+S[i]))

        A[i+1] = A[i]+tau*(beta*(a[i]-A[i]))
        B[i+1] = B[i]+tau*(beta*(b[i]-B[i]))
        C[i+1] = C[i]+tau*(beta*(c[i]-C[i]))

        S[i+1] = S[i]+tau*((-k_s0*S[i])+(k_s1*A[i])-(eta*(S[i]-Se[i])))

        if i not in S_dict.keys():
            S_dict[i] = [float(S[i])]
        else :
            S_dict[i].append(float(S[i]))

        Se[i] = (Q*np.mean(S_dict[i]))

    amplitude_a = np.transpose(a)
    amplitude_b = np.transpose(b)
    amplitude_c = np.transpose(c)
    amplitude_S = np.transpose(S)
    amplitude_Se = np.transpose(Se)

    #plt.plot(time,amplitude_a,label="a[i]")
    #plt.plot(time,amplitude_b)
    #plt.plot(time,amplitude_c, label="c[i]")
    #plt.plot(time, amplitude_Se, label="Se[i]")
    #plt.plot(time,amplitude_S, label ="S[i]")

    T = 60/(t/len(b))
    a = np.abs(fft.rfft(b, n=b.size))
    a[0]=0
    freqs = fft.rfftfreq(b.size, d=1./T)
    freqs = np.divide(60,freqs)

    max_freq = freqs[np.argmax(a)]
    #print("Cell of interest : " + str(cells))
    #print("Peak found at %s second period (%s minutes)" % (max_freq, max_freq/60))

    p = round(1/max_freq, 3)
    if p not in p_dict.keys():
        p_dict[p] = 1
    else:
        p_dict[p] +=1
#print(Se)
print(p_dict)
#plt.show()



##essais frequence des oscillation
##problemme : frequence en hertz par la frequence des oscillation
"""

spectrum = fft.fft(b)
freq = fft.fftfreq(len(spectrum))
plt.plot(freq, abs(spectrum))
plt.show()
print(freq)
"""

"""
print("###########################")

##confirmation des calculs sur petits echantillons
Se = np.zeros(10)
Se[0]=0

dic_test = {}
for cells in range(0,3):
    t = np.zeros(10)
    t[0] = 0
    beta = 2*cells
    for time in range (0,9):
        t[time+1] = t[time]+time+beta+Se[time]
        if time not in dic_test.keys():
            dic_test[time] = [int(t[time])]
        else :
            dic_test[time].append(int(t[time]))

        print(t[time])
        Se[time+1] = np.mean(dic_test[time])
    print("Valeurs de t pour chaque cellules : " + str(t))
print("Concentration de Se pour toutes les cellules : " + str(Se))
"""
