from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp



data = load_iris()



for i in range(len(data.data)):
    if data.target[i]==0:
        data.target[i]=-1
    if data.target[i]==2:
        data.target[i]=1
    if data.data[i,0]==4.5 and data.data[i,1]==2.3:
        data.target[i]=1
        
for i in range(len(data.data)):
    if data.target[i]==-1:
        plt.scatter(data.data[i,0],data.data[i,1], c="red")
    if data.target[i]==1:
        plt.scatter(data.data[i,0],data.data[i,1], c="blue")
 

n=len(data.data)    
X=np.array(data.data[:,0:2])
y=np.array(data.target)  
p=np.shape(X)[1]






alpha=cp.Variable(n)
alpha_constraint = [alpha[i] >=0 for i in range(n)]
alpha_y_constraint = [alpha@y.T == 0]
constraints=alpha_constraint+alpha_y_constraint

K = y[:,None]*X
K = np.dot(K,K.T)
obj = cp.Minimize(-cp.sum(alpha)+1/2*cp.quad_form(alpha,K))

prob = cp.Problem(obj, constraints)
prob.solve(solver='ECOS',verbose=True)  # Returns the optimal value.



# Calcul de w
w=np.zeros(p)
for i in range(n):
    w+=alpha.value[i]*y[i]*X[i]
    
# Calcul de b
b=0
for i in range(n):
    aux=(y[i]/np.abs(y[i])*(1/y[i]-np.dot(X[i],w)))
    b=max(b,aux)
b=-b

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


