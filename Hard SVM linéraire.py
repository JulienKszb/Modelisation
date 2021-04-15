import numpy as np
import matplotlib.pyplot as plt

# Jeu de données
x=np.array([[2,2],[3,4],[2,5],[6,6],[1,7],[5,7],[3,8],[7,9],[9,9],[5,10],[7,1],[11,1],[9,3],[14,3],[12,4],[15,4],[13,6],[15,7]])
n=np.shape(x)[0]
p=np.shape(x)[1]
# Classe des points
y=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1])

# Affichage jeu de données
plt.figure(1)
plt.scatter(x[:,0], x[:,1], c=y, cmap='bwr')
plt.show()

##############
def lagrangien(alpha): 
    somme=0
    som=0
    for i in range(n):
        som+=alpha[i]
        for j in range(n):
            somme+=alpha[i]*alpha[j]*y[i]*y[j]*np.dot(x[i],x[j])
    return (-1/2*somme + som) * (-1)

def jacobien(alpha):
    resultat=np.zeros(n)
    for k in range(n):
        somme = 0
        for i in range(n):
            somme += alpha[i]*y[i]*y[k]*np.dot(x[k],x[i])
        resultat[k]=1-somme
    return resultat*(-1)

##############
def descente_gradient(fonction,h=1e-3,alpha_initiale=np.zeros(n)):
    alpha = np.copy(alpha_initiale)
    y=[alpha]
    itermax = 10000
    iter = 0
    while iter<itermax:
        df = fonction(alpha)
        alpha = alpha - h*df
        for i in range(n):
            alpha[i]=np.maximum(alpha[i],0)
        y.append(alpha)
        iter += 1
    listalpha=np.array(y)
    return alpha,listalpha

alpha,listalpha=descente_gradient(jacobien)


# Calcul de w
w=np.zeros(p)
for i in range(n):
    w+=alpha[i]*y[i]*x[i]

# Calcul de b
b=0
for i in range(n):
    aux=(y[i]/np.abs(y[i])*(1/y[i]-np.dot(x[i],w)))
    b=max(b,aux)

print("w = ",w," et b = ",b)
print("alpha = ",alpha)

# Définition du plan séparateur
xx=np.linspace(0,15,1000)
#print(xx)

# On a w1*x1+w2*x2+b=0
yy=(-b-w[0]*xx)/w[1]
xx0=0
yy0=(-b-w[0]*xx0)/w[1]
xx1=15
yy1=(-b-w[0]*xx1)/w[1]

plt.figure(2)
plt.scatter(x[:,0],x[:,1],c=y,cmap='bwr')
plt.plot([xx0, xx1], [yy0, yy1], 'g--', lw=2)
plt.show()

