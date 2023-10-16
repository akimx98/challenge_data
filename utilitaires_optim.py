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
    

    x = theta
    y = [e(z) for z in x]
    x,y = np.stack(x),np.stack(y)
    dx,dy = (x[1:]-x[:-1])/2,(y[1:]-y[:-1])/2
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(x_grid,y_grid,color='black',label=r'$e(\theta)$')
    ax.plot(x,y,marker='o',color='red',label = 'Descente de dérivée')
    ax.arrow(x[0],y[0],dx[0],dy[0],head_width=0.07,overhang=3/5,head_length=0.07,color='red')

    ax.set_xlabel(r'$\theta$')
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