"""
BS1 project

retry all figures of article
repressilator, transpose idea : http://be150.caltech.edu/2020/content/lessons/08_repressilator.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random

##constantes
n = 2
alpha = 216
kappa = 20
k_s0 = 1
k_s1 = 0.01
t = 600
eta = 2.0
Q = 0.8

tau = 0.4 ##a voir

nb_cells = 10
Se = np.zeros(t)
Se[0] = 0

S_dict={}
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
        Se[i+1] = np.mean(S_dict[i])

    amplitude_a = np.transpose(a)
    amplitude_b = np.transpose(b)
    amplitude_c = np.transpose(c)
    amplitude_S = np.transpose(S)
    amplitude_Se = np.transpose(Se)

    #plt.plot(time,amplitude_a,label="a[i]")
    plt.plot(time,amplitude_b)
    #plt.plot(time,amplitude_c, label="c[i]")
    #plt.plot(time, amplitude_Se, label="Se[i]")
    #plt.plot(time,amplitude_S, label ="S[i]")
#print(Se)


plt.show()

"""
essais des histogrammes deltaB/beta
histogramme a comprendre

import collections
dic = {}
for i in range(0,1000):
    beta = random.gauss(1,0.05)
    t = (0.05/beta)
    t = round(t,3)
    if t not in dic.keys():
        dic[t] = 1
    else :
        dic[t] += 1

dic = od = collections.OrderedDict(sorted(dic.items()))
print(dic)
print(dic.keys())
plt.bar([ str(i) for i in dic.keys()], dic.values())
plt.show()
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
