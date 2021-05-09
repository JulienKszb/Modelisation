import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm

# Jeu de données
X=np.array([[9,8],[9,10],[10,7],[10,10],[11,9],[12,6],[12,8],[12,10],[13,6],[13,8],[13,9],[13,11],[14,3],[14,5],[14,8],[14,9],[15,7],[15,10],[16,4],[1,1],[1,4],[2,3],[2,5],[3,1],[3,4],[3,7],[4,2],[4,4],[4,6],[5,2],[5,3],[5,4],[6,1],[6,3],[6,5],[7,2],[7,3],[10,1]])
# Modification des données pour avoir des valeurs "cohérentes"
X[:,0]=X[:,0]*1.7+150
X[:,1]=X[:,1]*3+50

# Classe des points
y=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

###############
model = svm.SVC(kernel='linear', C=1000)
model.fit(X, y)



ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


xx = np.linspace(xlim[0]-4, xlim[1]+1, 50)
yy = np.linspace(ylim[0]-4, ylim[1]+1, 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)


ax.contour(XX, YY, Z, colors='k', levels=[-1,0,+1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')
plt.show()


fig, ax = plt.subplots()

cmap = sns.cubehelix_palette(start=-1, light=1, as_cmap=True)
sns.kdeplot(X[:19,0],X[:19,1],cmap, fill=True, ax=ax,)
sns.scatterplot(X[:19,0],X[:19,1], label='Homme')

cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.kdeplot(x=X[19:,0], y=X[19:,1],cmap=cmap, fill=True, ax=ax,)
sns.scatterplot(X[19:,0],X[19:,1],label='Femme')

ax.contour(XX, YY, Z, colors='red', levels=[0], alpha=0.7,linestyles='-')
ax.set_ylabel('Poids en kg')
ax.set_xlabel('Taille en cm')
ax.legend(loc='upper left', frameon=False)
#fig.savefig('SVM homme femme poids taille', dpi=300)
plt.show()





