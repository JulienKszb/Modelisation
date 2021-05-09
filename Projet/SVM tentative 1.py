import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl

echantillon=np.array([[2,2,-1],[3,4,-1],[2,5,-1],[6,6,-1],[1,7,-1],[5,7,-1],[3,8,-1],[7,9,-1],[9,9,-1],[5,10,-1],[7,1,1],[11,1,1],[9,3,1],[14,3,1],[12,4,1],[15,4,1],[13,6,1],[15,7,1]],dtype=object)

# Notre échantillon est constitué de n points qui sont dans Rd

def y(i):
    return echantillon[i][2]

def x(i):
    return echantillon[i][:2]

n=len(echantillon)
def afficher():
    for i in range(n):
        if y(i)==1:
            plt.scatter(x(i)[0], x(i)[1], c="red" )
        elif y(i)==-1:
            plt.scatter(x(i)[0], x(i)[1], c="blue" )
#afficher()
        





def lagrangien(alpha): 
    aux1=np.zeros(2)
    aux2=0
    for i in range(n):
        aux1+=np.dot(y(i)*alpha[i],x(i))
        aux2+=alpha[i]
    return 1/2*npl.norm(aux1,ord=2)**2-aux2


def gradient(alpha):
    resultat=np.zeros(n)
    for k in range(n):
        aux=0;
        for i in range(n):
            aux+=alpha[i]*y(i)*y(k)*np.dot(x(k),x(i))
            #alpha à deux 
        resultat[k]=aux-1
    return resultat

def fonction(alpha):
    return gradient(alpha)


def Gradient(function,h=1e-3,aini=np.zeros(n)):
    a = np.copy(aini)
    y=[a]
    eps = 1e-10
    itermax = 100000
    err = 2*eps
    iter = 0
    while err>eps and iter<itermax:
        df = function(a)
        a = a - h*df
        y.append(a)
        err = np.linalg.norm(df)
        iter += 1
    aiter=np.array(y)
    return a,aiter,iter


alpha,aiter,iter=Gradient(fonction)

#h=10**(-3)
#alpha=np.zeros(n)
#erreur=1
#while erreur>10**(-3):
#    temp=alpha-np.dot(h,gradient(alpha))
#    temp=np.maximum(0,temp)
#    erreur=abs(npl.norm((alpha-temp),ord=2))/abs(npl.norm(alpha,ord=2))
    #print(erreur)
#    alpha=temp
    #print(alpha)

#print("alpha = " , alpha)
print(iter)

w=np.zeros(2)
for j in range(n):
    w=np.dot(alpha[j]*y(j),x(j))
print("w = ", w)        
            













plt.figure(3)
for i in range(n):
    if y(i)==1:
        plt.scatter(x(i)[0], x(i)[1], c="red" )
    elif y(i)==-1:
            plt.scatter(x(i)[0], x(i)[1], c="blue" )
plt.scatter(w[0],w[1], c="green")

# On chercher  à minimiser fonction pour obtenir le vecteur alpha, qui sert a déterminer w et b.


