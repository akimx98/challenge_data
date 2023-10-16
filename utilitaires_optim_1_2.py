## Notebook 1

import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

def descente(theta_0,alpha,N):
    a = 1
    b = 2
    c = 3

    #dérivée de f
    def df(x):
        return 2*a*x+b
    
    #Fonction Etape
    def étape(theta):
        return theta-alpha*df(theta)
    
    #On sauvegarde les valeurs successives de x
    theta = []
    theta.append(theta_0)
    for n in range(N):
        theta_0 = étape(theta_0)
        theta.append(theta_0)
    return(theta)

def affichage_descente(theta):
    a = 1
    b = 2
    c= 3
    def e(x):
        return a*x**2+b*x+c
    
    theta = np.stack(theta)
    minus = min(-1,theta.min())-0.25
    maxus = max(-1,theta.max())+0.25
    x_grid = np.linspace(minus,maxus,1000)

    y_grid = [e(x_grid[i]) for i in range(1000)]

    x_grid,y_grid = np.stack(x_grid),np.stack(y_grid)
    plt.plot(x_grid,y_grid,color='black',label=r'$e(\theta)$')

    x= theta
    y = [e(z) for z in x]
    x,y = np.stack(x),np.stack(y)
    dx,dy = (x[1:]-x[:-1])/2,(y[1:]-y[:-1])/2
    fig, ax = plt.figure(figsize=(6,5))
    ax.plot(x,y,marker='o',color='red',label = 'Descente de dérivée')
    ax.arrow(x[0],y[0],dx[0],dy[0],head_width=0.07,overhang=3/5,head_length=0.07,color='red')

    ax.xlabel(r'$\theta$')
    ax.legend()
    plt.show()

def erreur_MNIST(theta):
    #A completer
    return #theta**2-8*theta

def d_erreur_MNIST(theta):
    #A completer
    return #2*theta-8

def affichage_MNIST(theta):
    X = np.arange(len(theta))
    Y = [erreur_MNIST(theta[i]) for i in range(len(theta))]

    plt.plot(X,Y,marker ='o',label=r'$e(\theta_n)$')
    plt.legend(fontsize=20)
    plt.xlabel(r'$n$',fontsize=20)

def descente_MNIST(theta,alpha,N):
    theta = descente(theta_0,alpha,N,d_erreur_MNIST)
    affichage_MNIST(theta)
    return theta[-1]

###A Mettre dans un package
def approx_derivee(eta):
    N =1000
    n_df = 590

    x = np.linspace(0,2*np.pi,N)

    f = np.cos(x)+x**3/1e2

    df = -np.sin(x[n_df])+3*x[n_df]**2/1e2

    n_eta = int((eta/(2*np.pi/N)))+1
    df_eta = (f[n_df+n_eta] - f[n_df])/(x[n_df+n_eta]-x[n_df])

    plt.plot(x,f,color='black',label=r'$e(\theta)$' )
    plt.plot(x,(x-x[n_df])*df+f[n_df],label=r'$e(\theta_0)+ e\'(\theta_0)(\theta-\theta_0 )$',color = 'green')
    plt.plot(x,(x-x[n_df])*df_eta+f[n_df],label=r'$y(\theta)$',color='orange')
    plt.scatter(x[n_df],f[n_df],marker='o',color='orange')
    plt.scatter(x[n_df+n_eta],f[n_df+n_eta],marker='o',color='orange')

    #plt.plot([0,x[n_df]],[f[n_df],f[n_df]],c='black',linestyle='dashed')
    plt.xticks([x[n_df],x[n_df+n_eta]],labels=[r'$\theta_0}$',r'$\theta_0+\eta$'],fontsize=10)
    plt.yticks([f[n_df],f[n_df+n_eta]],labels=[r'$f(\theta_0})$',r'$f(\theta_0+\eta)$'],fontsize=10)

    plt.legend(fontsize=12)
    plt.xlabel(r'$\theta$',fontsize=20)
    plt.show()

## Notebook 2

# Import des librairies utilisées dans le notebook
#import basthon
import requests
import numpy as np
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO


plt.rcParams['figure.dpi'] = 150

# Téléchargement et extraction des inputs contenus dans l'archive zip
inputs_zip_url = "https://raw.githubusercontent.com/challengedata/challenge_educatif_mnist/main/inputs.zip"
inputs_zip = requests.get(inputs_zip_url)
zf = ZipFile(BytesIO(inputs_zip.content))
zf.extractall()
zf.close()


# Téléchargement des outputs d'entraînement de MNIST-10 contenus dans le fichier y_train_10.csv
output_train_url = "https://raw.githubusercontent.com/challengedata/challenge_educatif_mnist/main/y_train_10.csv"
output_train = requests.get(output_train_url)

# Création des variables d'inputs, outputs et indices pour les datasets MNIST-2, MNIST-4 et MNIST-10

# MNIST-10

# Inputs and indices
with open('mnist_10_x_train.pickle', 'rb') as f:
    ID_train_10, x_train_10 = pickle.load(f).values()

with open('mnist_10_x_test.pickle', 'rb') as f:
    ID_test_10, x_test_10 = pickle.load(f).values()

# Outputs
_, y_train_10 = [np.loadtxt(StringIO(output_train.content.decode('utf-8')),
                                dtype=int, delimiter=',')[:,k] for k in [0,1]]

# MNIST-2

chiffre_1 = 1
chiffre_2 = 6
chiffres = [chiffre_1, chiffre_2]

# Trouver les indices des étiquettes qui valent 0 ou 1
indices = np.where((y_train_10 == chiffre_1) | (y_train_10 == chiffre_2))

# Utiliser ces indices pour extraire les images correspondantes de x_train_10
x_train = x_train_10[indices]
y_train = y_train_10[indices]

# classe : -1/1
y_train[y_train == 1] = -1
y_train[y_train == 6] = 1

N = len(x_train)

def affichage(seuil, poids, X, Y):
    # plot the line, the points, and the nearest vectors to the plane
    xx = np.linspace(min(X[:,0])-.5, max(X[:,0])+.5, 10)
    yy = np.linspace(min(X[:,1])-.5, max(X[:,1])+.5, 10)
    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = estimateur(seuil, poids, np.array([[x1, x2]]))
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ["dashed", "solid", "dashed"]
    colors = "k"
    plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolor="black", s=20)

    plt.axis("tight")
    plt.show()

def estimateur(seuil, poids, x):
    return (np.matmul(x, poids)-seuil).reshape(x.shape[0])
    
def erreur(y_pred, y_vrai):
    return np.exp(-y_vrai*y_pred)#np.minimum(np.exp(-y_vrai*y_pred),1)#

def d_erreur(y_pred, y_vrai):
    return -erreur(y_pred, y_vrai)

def descente_simultanee(seuil_0, poids_0, X, y_vrai, pas = .1, nb_etapes = 100, plot = False):
    seuil = seuil_0
    poids = poids_0.copy()
    evolution_erreur = []
    
    for n in range(nb_etapes):
        y_pred = estimateur(seuil, poids, X)
        seuil -= pas*np.mean(-y_vrai*d_erreur(y_pred, y_vrai))
        poids -= pas*np.mean((y_vrai*d_erreur(y_pred, y_vrai)).reshape(X.shape[0],1)*X, axis=0).reshape(poids.shape)
        evolution_erreur.append(np.mean(erreur(y_pred, y_vrai)))
    
    if plot:
        plt.plot(evolution_erreur, label='l')
        plt.legend(loc='best')
        plt.xlabel("Nombre d'époques")
        plt.ylabel('Erreur')
        plt.show()
        
    return seuil, poids