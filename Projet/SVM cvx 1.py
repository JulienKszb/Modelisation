import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Jeu de données
X=np.array([[9,8],[9,10],[10,7],[10,10],[11,9],[12,6],[12,8],[12,10],[13,6],[13,8],[13,9],[13,11],[14,3],[14,5],[14,8],[14,9],[15,7],[15,10],[16,4],[1,1],[1,4],[2,3],[2,5],[3,1],[3,4],[3,7],[4,2],[4,4],[4,6],[5,2],[5,3],[5,4],[6,1],[6,3],[6,5],[7,2],[7,3],[10,1]])
# Modification des données pour avoir des valeurs "cohérentes"
X[:,0]=X[:,0]*1.7+150
X[:,1]=X[:,1]*3+50

# Classe des points
y=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

n=np.shape(X)[0]
p=np.shape(X)[1]




alpha=cp.Variable(n)
alpha_constraint = [alpha[i] >=0 for i in range(n)]
alpha_y_constraint = [alpha@y.T == 0]
constraints=alpha_constraint+alpha_y_constraint

K = y[:,None]*X
K = np.dot(K,K.T)
obj = cp.Minimize(-cp.sum(alpha)+1/2*cp.quad_form(alpha,K))

prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.



# Calcul de w
w=np.zeros(p)
for i in range(n):
    w+=alpha.value[i]*y[i]*X[i]
    
# Calcul de b
b=0
for i in range(n):
    aux=(y[i]/np.abs(y[i])*(1/y[i]-np.dot(X[i],w)))
    b=max(b,aux)



print("w = ",w," et b = ",b)

# Définition du plan séparateur
xx=np.linspace(0,15,1000)
#print(xx)

# On a w1*x1+w2*x2+b=0
yy=(-b-w[0]*xx)/w[1]
xx0=np.amin(X[:,0])
yy0=(-b-w[0]*xx0)/w[1]
xx1=np.amax(X[:,0])
yy1=(-b-w[0]*xx1)/w[1]


print(xx0,xx1)
plt.figure(2)
plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
plt.plot([xx0, xx1], [yy0, yy1], 'g--', lw=2)
plt.show()


