"""
BS1 project

retry all figures of article
repressilator, transpose idea : http://be150.caltech.edu/2020/content/lessons/08_repressilator.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random



for cells in range(0,10):
    ##constantes
    n = 2
    alpha = 216
    kappa = 20
    k_s0 = 1
    k_s1 = 0.01
    t = 600
    eta = 2.0
    Q = 0.8
    beta = random.gauss(1,0.05)
    tau = 0.4 ##a voir

    ##initialisation des matrices
    a = np.zeros(t)
    b = np.zeros(t)
    c = np.zeros(t)

    A = np.zeros(t)
    B = np.zeros(t)
    C = np.zeros(t)

    S = np.zeros(t)
    Se = np.zeros(t)

    a[0] = 0
    b[0] = 0
    c[0] = 0

    A[0] = 0
    B[0] = 0
    C[0] = 0

    S[0] = 0
    Se[0] = 0


    time = np.arange(0,t)
    for i in range(0,t-1):
        a[i+1] = a[i] + tau *(-a[i]+(alpha/(1+C[i]**n)))
        b[i+1] = b[i] + tau *(-b[i]+(alpha/(1+A[i]**n)))
        c[i+1] = c[i] + tau *(-c[i]+(alpha/(1+B[i]**n))+(kappa*S[i]/1+S[i]))

        A[i+1] = A[i]+tau*(beta*(a[i]-A[i]))
        B[i+1] = B[i]+tau*(beta*(b[i]-B[i]))
        C[i+1] = C[i]+tau*(beta*(c[i]-C[i]))

        S[i+1] = S[i]+tau*((-k_s0*S[i])+(k_s1*A[i])-(eta*(S[i]-Se[i])))

        Se[i+1] = Q*np.mean(S[0:i+1])

    #print(Se)
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

#plt.show()

"""
essais des histogrammes deltaB/beta
"""
dic = {}
for i in range(0,1000):
    beta = random.gauss(1,0.05)
    t = (0.05/beta)
    t = round(t,3)
    if t not in dic.keys():
        dic[t] = 1
    else :
        dic[t] += 1

print(dic)
print(dic.keys())
plt.bar([ str(i) for i in dic.keys()], dic.values())
plt.show()
