from utilitaires_mnist_2 import *
from IPython.display import display # Pour afficher des DataFrames avec display(df)
import pandas as pd
# import mplcursors

start_analytics_session(2)

try:
    # For dev environment
    from strings_geo import *
except ModuleNotFoundError: 
    pass

def affichage_2_geo(display_k=False):
    try:
        zone_1 = get_variable('zone_1')
        zone_2 = get_variable('zone_2')
        deux_caracteristiques = get_variable('deux_caracteristiques')
    except NameError:
        print_error("Erreur innatendue.")
        return

    x2 = d
    x7 = d_train[2,:,:].copy()
    images = np.array([x7, x2])
    fig, ax = plt.subplots(1, len(images), figsize=(8, 2))
    c_train = np.array([np.array(deux_caracteristiques(d)) for d in images])
    
    for i in range(len(images)):
        imshow(ax[i], images[i])
        k = c_train[i]
        if i == 0:
            zoneNamePos = 'left'
        elif i == len(images) - 1:
            zoneNamePos = 'right'
        else:
            zoneNamePos = 'center'
        
        k1 = f'{k[0]:.2f}' if display_k else '?'
        k2 = f'{k[1]:.2f}' if display_k else '?'
        outline_selected(ax[i], zone_1[0], zone_1[1], zoneName=f'$k_1 = {k1}$', zoneNamePos=zoneNamePos)
        outline_selected(ax[i], zone_2[0], zone_2[1], zoneName=f'$k_2 = {k2}$', zoneNamePos=zoneNamePos)

    plt.show()
    plt.close()

    if not display_k:
        df = pd.DataFrame({'$k_1$': c_train[:,0], '$k_2$': c_train[:,1], '$r$': ['$r_1$ = 2 ou 7 ?', '$r_2$ = 2 ou 7 ?']})
        df.index += 1
        display(df)
        return

def affichage_dix_2():
    try:
        zone_1 = get_variable('zone_1')
        zone_2 = get_variable('zone_2')
        deux_caracteristiques = get_variable('deux_caracteristiques')
    except NameError:
        print_error("Exécute la cellule précédente pour définir la fonction caracteristique.")
        return
    
    display_zone_1 = zone_1 + ['k1']
    display_zone_2 = zone_2 + ['k2']
    affichage_dix(d_train, liste_y=None, zones=[display_zone_1, display_zone_2])

    c_train = [deux_caracteristiques(d) for d in d_train[0:10]]
    df = pd.DataFrame()
    df['$r$ (label)'] = r_train[0:10]   
    df['$k_1$'] = [k[0] for k in c_train]
    df['$k_2$'] = [k[1] for k in c_train]
    df.index+=1
    display(df)
    return


def affichage_2_cara(image1, image2, A1=None, B1=None, A2=None, B2=None, displayPoints=False, titre1="", titre2=""):
    """Fonction qui affiche deux images côte à côté, avec un rectangle rouge délimité par les points A et B
        A : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        B : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
    Si displayPoints est True, les points A et B sont affichés
    """
    if image1.min().min() < 0 or image1.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels des images doivent être compris entre 0 et 255.")
        return
    
    if image2.min().min() < 0 or image2.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels des images doivent être compris entre 0 et 255.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    imshow(ax[0], image1)
    ax[0].set_title(titre1)
    ax[0].set_xticks(np.arange(0,28,5))
    ax[0].xaxis.tick_top()
    outline_selected(ax[0], A1, B1, displayPoints, nameA='A1', nameB='B1', color='red')
    outline_selected(ax[0], A2, B2, displayPoints, nameA='A2', nameB='B2', color='C0')

    imshow(ax[1], image2)
    ax[1].set_title(titre2)
    ax[1].set_xticks(np.arange(0,28,5))
    ax[1].xaxis.tick_top()
    outline_selected(ax[1], A1, B1, displayPoints, nameA='A1', nameB='B1', color='red')
    outline_selected(ax[1], A2, B2, displayPoints, nameA='A2', nameB='B2', color='C0')

    plt.show()
    plt.close()
    
    affichage_10_tableau_deux_caracteristiques(A1=A1, B1=B1, A2=A2, B2=B2)
    return


def affichage_10_tableau_deux_caracteristiques(A1=None, B1=None, A2=None, B2=None):
    """Fonction qui affiche une frise de 10 images et le tableau des caractéristiques de ces 10 images"""

    deux_caracteristiques = get_variable('deux_caracteristiques')

    images = d_train[0:10]
    labels = r_train[0:10]
    c_train = [deux_caracteristiques(d) for d in images]

    affichage_dix_2(images, liste_y=labels, A1=A1, B1=B1, A2=A2, B2=B2)
    
    df = pd.DataFrame()
    df['$r$ (classe)'] = labels   
    df['$k_1$'] = [k[0] for k in c_train]
    df['$k_2$'] = [k[1] for k in c_train]
    #df['$mk_1 + p$'] = [m * k[0] + p for k in c_train]
    #df['$\hat{r}$ (prediction)'] = '?'
    df.index+=1
    display(df)
    return

def affichage_dix_2(images, A1=None, B1=None, A2=None, B2=None, liste_y = r_train, n=10):
    global r_train
    fig, ax = plt.subplots(1, n, figsize=(n, 1))
    
    # Cachez les axes des subplots
    for j in range(n):
        ax[j].axis('off')
        imshow(ax[j], images[j])
        outline_selected(ax[j], A1, B1, color='red')
        outline_selected(ax[j], A2, B2, color='C0')
    
    # Affichez les classes
    if liste_y is not None:
        for k in range(n):
            fig.text((k+0.5)/10, 0, '$r = $'+str(liste_y[k]), va='top', ha='center', fontsize=12)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.2, wspace=0.05, hspace=0)
    plt.show()
    plt.close()

def compute_c_train(images=d_train, labels=r_train):
    try:
        deux_caracteristiques = get_variable('deux_caracteristiques')
    except NameError:
        print_error("Exécute la cellule précédente pour définir la fonction deux_caracteristiques")
        return
    
    c_train = np.array([deux_caracteristiques(d) for d in images])
    return c_train

def compute_c_train_by_class(images=d_train, labels=r_train, c_train=None):
    if c_train is None:
        c_train = compute_c_train(images, labels)
    c_train_par_population = [c_train[labels==k] for k in classes]
    return c_train_par_population


def create_graph(figsize=(6,6)):
    fig, ax = plt.subplots(figsize=figsize)

    # Enlever les axes de droites et du haut
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Centrer les axes en (0,0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(("data", 0))

    
    #Afficher les flèches au bout des axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)   
    
    # Nom des axex
    ax.set_xlabel('$k_1$', loc='right')
    ax.set_ylabel('$k_2$', loc='top', rotation='horizontal')

    return fig, ax

        # @cursor.connect("add")
        # def on_add(sel):
        #     sel.annotation.set_text("")  # Remove the default annotation
        #     i = sel.index  # Get the index of the point
        #     displayed_images[i].set(alpha=1)
        #     nonlocal displayed_image, fig
        #     print('add')
        #     print(displayed_image)
        #     displayed_image = ax.imshow(images[sel.index], cmap='gray', vmin=0, vmax=255, extent=(x[sel.index]-11, x[sel.index]-1, y[sel.index]-11, y[sel.index]-1))
        #     plt.show()


        # @cursor.connect("remove")
        # def on_remove(sel):
        #     i = sel.index  # Get the index of the point
        #     displayed_images[i].set(alpha=0.5)
        #     nonlocal displayed_image, fig
        #     print('remove')
        #     print(displayed_image)
        #     if displayed_image is not None:
        #         displayed_image.remove()
        #         displayed_image = None

        # def on_move(event):
        #     nonlocal displayed_image
        #     if event.xdata and event.ydata and scatter.contains(event)[0]:
        #         i = scatter.contains(event)[1]['ind'][0]
        #         if displayed_image is None:
        #             displayed_image = ax.imshow(images[i], cmap='gray', vmin=0, vmax=255, extent=(x[i]-11, x[i]-1, y[i]-11, y[i]-1))
        #             plt.show()

        #     elif displayed_image is not None and not (event.xdata and event.ydata and scatter.contains(event)[0]):
        #         displayed_image.remove()
        #         displayed_image = None
        #         plt.show()

        # fig.canvas.mpl_connect('motion_notify_event', on_move)

def tracer_droite(ax, m, p, x_min, x_max, color='black'):
    # Ajouter la droite
    x = np.linspace(x_min, x_max, 1000)
    y = m*x + p
    ax.plot(x, y, c=color)  # Ajout de la droite en noir

    # Calculate a point along the line
    x_text = x_max - 2
    y_text = m*x_text + p - 4
    if y_text > x_max - 2:
        y_text = x_max - 2
        x_text = (y_text - p)/m + 4

    # Calculate the angle of the line
    angle = np.arctan(m) * 180 / np.pi

    # Display the equation of the line
    equation = f'$k_2 = {m}k_1 + {p}$'
    ax.text(x_text, y_text, equation, rotation=angle, color=color, verticalalignment='top', horizontalalignment='right')

def tracer_2_points():
    try:
        deux_caracteristiques = get_variable('deux_caracteristiques')
    except NameError:
        print_error("Exécute la cellule précédente pour définir la fonction deux_caracteristiques")
        return

    images = d_train[0:2]
    c_train = [deux_caracteristiques(d) for d in images]
    r_train_loc = r_train[0:2]

    x = [p[0] for p in c_train]
    y = [p[1] for p in c_train]

    fig, ax = create_graph(figsize=(5,5))

    ax.scatter(x, y, marker = '+', c='black', zorder=2)

    labels = ['A', 'B']
    for i in range(2):
        ax.annotate(labels[i], (x[i] + 0.5, y[i] + 0.5), zorder=1)
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=255, extent=(x[i]-11, x[i]-1, y[i]-11, y[i]-1), zorder = 1)
        

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))
    
    ax.set_axisbelow(True)
    # Ajout d'une grille en pointillés
    ax.grid(True, linestyle='--', zorder=0)
    
    plt.show()
    plt.close()

    df = pd.DataFrame()
    labels = ['Point A :', 'Point B :']
    df.index = labels
    #df.index.name = 'Point'
    df['$r$'] = ['$7$', '$2$']
    df['$k_1$'] = ['$?$', '$?$']
    df['$k_2$'] = ['$?$', '$?$']
    display(df)
    return


def tracer_200_points(nb=200):
    c_train_par_population = compute_c_train_by_class(d_train[0:nb], r_train[0:nb])

    # Deux premières couleurs par défaut de Matplotlib
    nb_digits = 2
    colors = ['C0', 'C1']
     
    fig, ax = create_graph()
    for i in range(nb_digits):  # ordre inversé pour un meilleur rendu
        ax.scatter(c_train_par_population[i][:,0], c_train_par_population[i][:,1], marker = '+', s = 20, c=colors[i])

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    x_min, x_max = min(0, mins_[0], mins_[1]), max(0, maxs_[0] + 2, maxs_[1] + 2)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    patches = [mpatches.Patch(color=colors[i], label="$r = ?$") for i in range(nb_digits)]
    ax.legend(handles=patches,loc='upper left')
    
    plt.show()
    plt.close()

def tracer_10_points_droite():
    images = d_train[20:30]
    labels = r_train[20:30]

    affichage_dix(images, liste_y=labels)

    c_train_par_population = compute_c_train_by_class(images, labels)

    fig, ax = create_graph()

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    x_min, x_max = min(0, mins_[0], mins_[1]), max(0, maxs_[0] + 2, maxs_[1] + 2)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    # Deux premières couleurs par défaut de Matplotlib
    colors = ['C0', 'C1']

    # Add colored areas
    x = np.linspace(x_min, x_max, 1000)
    y = 0.5*x + 20
    ax.fill_between(x, y, x_max, where=(y < x_max), facecolor=colors[0], alpha=0.3, label='$\hat r = 2$')
    ax.fill_between(x, y, x_min, where=(y > x_min), facecolor=colors[1], alpha=0.3, label='$\hat r = 7$')

    tracer_droite(ax, 0.5, 20, x_min, x_max)

    for i in range(len(classes)):  # ordre inversé pour un meilleur rendu
        ax.scatter(c_train_par_population[i][:,0], c_train_par_population[i][:,1], marker = '+', s = 20, c=colors[i])


    patches = [mpatches.Patch(color=colors[i], label=f"$r = {classes[i]}$") for i in range(len(classes))] + \
                [mpatches.Patch(color=colors[i], alpha=0.3, label=f"$\hat r = {classes[i]}$") for i in range(len(classes))]
    ax.legend(handles=patches,loc='upper left')
    
    plt.show()
    plt.close()

def schema_droite_areas():
    # Display a straight line with colored areas on each side
    fig, ax = create_graph()
    tracer_droite(ax, 1, 0, -10, 10, color='black')
    
    
def tracer_separatrice(m, p, display_misses=False):
    """Tracer les points et la droite séparatrice
    m : float, coefficient directeur de la droite
    p : float, ordonnée à l'origine de la droite
    display_misses : bool, afficher les points mal classés"""

    if not (isinstance(m, (int, float)) and isinstance(p, (int, float))):
        print_error("Les paramètres m et p doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return
    
    c_train = compute_c_train()
    c_train_par_population = compute_c_train_by_class(c_train=c_train)

    r_est_train_par_population = [_classification(m, p, k) for k in c_train_par_population]

    error = erreur_lineaire(m, p, c_train)
    print(f"Pourcentage d'erreur : {error:.2f}%")

    colors=['C0', 'C1']
    nb_digits = 2

    fig, ax = create_graph()

    if display_misses:
        for i in range(nb_digits):
            points = c_train_par_population[i][r_est_train_par_population[i] == classes[i]]
            ax.scatter(points[:,0], points[:,1], marker = '+', s = 20, c=colors[i], linewidths=0.5)
        
        # On finit par les points mal classés pour qu'ils soient visibles
        for i in range(nb_digits):
            points = c_train_par_population[i][r_est_train_par_population[i] != classes[i]]
            ax.scatter(points[:,0], points[:,1], marker = '+', s = 20, c=colors[i], linewidths=0.5)
    else:
        ax.scatter(c_train[:,0], c_train[:,1], marker = '+', s = 20, c=['C0' if r == 2 else 'C1' for r in r_train], linewidths=0.5)
    
    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    x_min, x_max = min(0, mins_[0], mins_[1]), max(0, maxs_[0] + 2, maxs_[1] + 2)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    tracer_droite(ax, m, p, x_min, x_max)

    patches = [mpatches.Patch(color=colors[i], label=f"$r = {classes[i]}$") for i in range(nb_digits)]
    ax.legend(handles=patches,loc='upper left')
    
    plt.show()
    plt.close() 

    if (error < 8):
        print("Bravo ! Vous pouvez passer à la suite du notebook ou continuer à chercher la droite optimale.")
        validation_question_4()
    else:
        print(f"Vous avez un taux d'erreur de {error:.2f}%. Continuer à chercher la droite optimale en changeant m et p, pour avoir moins de 8%.")

        
def tracer_separatrice_2(m, p):
    """Tracer les points et la droite séparatrice
    m : float, coefficient directeur de la droite
    p : float, ordonnée à l'origine de la droite
    display_misses : bool, afficher les points mal classés"""

    if not (isinstance(m, (int, float)) and isinstance(p, (int, float))):
        print_error("Les paramètres m et p doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return
    
    c_train = compute_c_train()
    c_train_par_population = compute_c_train_by_class(c_train=c_train)


    error = erreur_lineaire(m, p, c_train)
    if error > 50:
        error = 100 - error

    print(f"Pourcentage d'erreur : {error:.2f}%")


    colors=['C0', 'C1']
    nb_digits = 2

    fig, ax = create_graph()

    ax.scatter(c_train[:,0], c_train[:,1], marker = '+', s = 20, c=['C0' if r == 2 else 'C1' for r in r_train], linewidths=0.5)
    
    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    x_min, x_max = min(0, mins_[0], mins_[1]), max(0, maxs_[0] + 2, maxs_[1] + 2)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    tracer_droite(ax, m, p, x_min, x_max)

    patches = [mpatches.Patch(color=colors[i], label=f"$r = {classes[i]}$") for i in range(nb_digits)]
    ax.legend(handles=patches,loc='upper left')
    
    plt.show()
    plt.close() 

def tracer_point_droite():
    global g_m, g_p
    deux_caracteristiques = get_variable('deux_caracteristiques')
    m = g_m
    p = g_p

    points = [(20, 40), (35,25)]

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    y += [m * k1 + p for k1 in x]
    x += x

    fig, ax = create_graph(figsize=(5,5))

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    ax.set_yticks(np.arange(x_min, x_max, 5))

    scatter = ax.scatter(x, y, marker = '+', c='black')

    labels = ['A', 'B', 'M', 'N']
    for i in range(len(labels)):
        ax.annotate(labels[i], (x[i] + 0.5, y[i] - 0.5), va='center')

        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i]/x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i]/x_max, linestyle='dotted', color='gray')

        if i >= len(points):
            # Annotate the y-axis with the y value
            ax.annotate(f"${labels[i]}k_2 = ?$", (0, y[i]), textcoords="offset points", xytext=(-25,0), ha='right', va='center')
        
    tracer_droite(ax, m, p, x_min, x_max)
    
    plt.show()
    plt.close()
    
pointA = (20, 40)

def tracer_exercice_classification(display_M_coords=False):
    global g_m, g_p
    deux_caracteristiques = get_variable('deux_caracteristiques')
    m = g_m
    p = g_p

    x = [pointA[0]]
    y = [pointA[1]]

    y += [m * k1 + p for k1 in x]
    x += x

    fig, ax = create_graph(figsize=(5,5))

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))
    
    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    ax.set_yticks(np.arange(x_min, x_max, 5))

    Mk2 = m * pointA[0] + p
    
    labels = [f'A({pointA[0]}, {pointA[1]})', f'M({pointA[0]}, {round(Mk2,2)})' if display_M_coords else 'M(?, ?)']
    colors = ['C4', 'C3']
    for i in range(len(labels)):
        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i]/x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i]/x_max, linestyle='dotted', color='gray')

        ax.annotate(labels[i], (x[i] + 1, y[i]), va='center', color=colors[i])
        ax.scatter(x[i], y[i], marker = '+', c=colors[i])

    tracer_droite(ax, m, p, x_min, x_max, color=colors[1])

    return ax
    
    
def tracer_point_droite():
    ax = tracer_exercice_classification()
    plt.show()
    plt.close()

def _classification(m, p, c_train):
    r_est_train = np.array([2 if k[1] > m*k[0]+p else 7 for k in c_train])
    return r_est_train
    
def erreur_lineaire(m, p, c_train):
    r_est_train = _classification(m, p, c_train)
    erreurs = (r_est_train != r_train).astype(int)
    return 100*np.mean(erreurs)


# VALIDATION

def setup_charts():
    # take only 1 on 2 images to speed up the display
    c_train_by_categories = compute_c_train_by_class(images=d_train[::2], labels=r_train[::2])
    points_2_str = ""
    for k in c_train_by_categories[0]:
        points_2_str += f"{{x: {k[0]}, y: {k[1]}}},"
    
    points_7_str = ""
    for k in c_train_by_categories[1]:
        points_7_str += f"{{x: {k[0]}, y: {k[1]}}},"
        
    js_code = f""" 
    const func = () => {{
        const min_x = 0
        const max_x = 80

        let chart;

        const slider_m = document.getElementById("slider_m")
        if (!slider_m) {{
            setTimeout(func, 1000)
            return
        }}
        const slider_p = document.getElementById("slider_p")
        const label_m = document.getElementById("label_m")
        const label_p = document.getElementById("label_p")

        let exec = null;
        
        const update = () => {{
            if (!chart) {{
                return
            }}

            const m = parseFloat(slider_m.value);
            const p = parseFloat(slider_p.value);

            const onScore = (data) => {{
                if (data.msg_type === "execute_result") {{
                    exec = null;
                    const error = Math.round(parseFloat(data.content.data['text/plain']) * 100) / 100 
                    const score = document.getElementById("score")
                    score.innerHTML = `${{error}}%`
                }}
            }}

            const python = `compute_score(${{m}}, ${{p}})`
            if (exec) {{
                clearTimeout(exec)
            }}

            exec = setTimeout(() => {{
                Jupyter.notebook.kernel.execute(python, {{
                    iopub: {{
                        output: onScore
                    }}
                }}, {{silent:false}})
            }}, 200)

            
            const data = [m * min_x + p, m * max_x + p]
            chart.data.datasets[0].data = data
            chart.data.datasets[0].label = `y = ${{m}}x + ${{p}}`

            chart.update('none')
            
            label_m.innerHTML =`m = ${{m}}`
            label_p.innerHTML = `p = ${{p}}`
        }}
        
        slider_m.addEventListener("input", update);
        slider_p.addEventListener("input", update);

        const onload = () => {{
            const ctx = document.getElementById('chart').getContext('2d');

            const data = {{
                labels: [min_x, max_x],
                datasets: [
                    {{
                        type: 'line',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        fill: false,
                        pointRadius: 0,  // Hides the points on the line
                    }},
                    {{
                        data: [{points_2_str}],
                        type: 'scatter',
                        borderColor: 'blue',
                        backgroundColor: 'blue',
                        pointRadius: 1,
                        pointHoverRadius: 3,
                        label: 'r = 2',
                        showLine: false,
                    }},
                    {{
                        data: [{points_7_str}],
                        type: 'scatter',
                        borderColor: 'orange',
                        backgroundColor: 'orange',
                        pointRadius: 1,
                        pointHoverRadius: 3,
                        label: 'r = 7',
                        showLine: false,
                    }},
                ]
            }};

            const config = {{
                data: data,
                options: {{
                    scales: {{
                        x: {{
                            type: 'linear',
                            //position: 'center',
                            min: min_x,   // Minimum value for x-axis
                            max: max_x   // Maximum value for x-axis
                        }},
                        y: {{
                            type: 'linear',
                            //position: 'center',
                            min: min_x,   // Minimum value for y-axis, adjust as necessary
                            max: max_x,   // Maximum value for y-axis, adjust as necessary
                        }}
                    }},
                }}
            }};

            chart = new Chart(ctx, config);
            update()
        }};

        window.mathadata.import_js_script("https://cdn.jsdelivr.net/npm/chart.js", onload)
    }}
    func()
    """

    run_js(js_code)

def setup_charts_2():
    # take only 1 on 2 images to speed up the display
    c_train_by_categories = compute_c_train_by_class(images=d_train[::2], labels=r_train[::2])
    max = np.max(np.concatenate(c_train_by_categories))
    points_2_str = ""
    for k in c_train_by_categories[0]:
        points_2_str += f"{{x: {k[0]}, y: {k[1]}}},"
    
    points_7_str = ""
    for k in c_train_by_categories[1]:
        points_7_str += f"{{x: {k[0]}, y: {k[1]}}},"
        
    js_code = f""" 
    const func = () => {{
        const min_x = 0
        const max_x = {max + 2}

        let chart;

        const slider_m = document.getElementById("input_m")
        if (!slider_m) {{
            setTimeout(func, 1000)
            return
        }}
        const slider_p = document.getElementById("input_p")

        let exec = null;
        
        const update = () => {{
            if (!chart) {{
                return
            }}

            const m = parseFloat(slider_m.value);
            const p = parseFloat(slider_p.value);

            const onScore = (data) => {{
                if (data.msg_type === "execute_result") {{
                    exec = null;
                    let error = parseFloat(data.content.data['text/plain'])
                    if (error > 50) {{
                        error = 100 - error
                    }}
                    error = Math.round(error * 100) / 100
                    const score = document.getElementById("score_custom")
                    score.innerHTML = `${{error}}%`
                }}
            }}

            const python = `compute_score(${{m}}, ${{p}})`
            if (exec) {{
                clearTimeout(exec)
            }}

            exec = setTimeout(() => {{
                Jupyter.notebook.kernel.execute(python, {{
                    iopub: {{
                        output: onScore
                    }}
                }}, {{silent:false}})
            }}, 200)

            
            const data = [m * min_x + p, m * max_x + p]
            chart.data.datasets[0].data = data
            chart.data.datasets[0].label = `y = ${{m}}x + ${{p}}`

            chart.update('none')
        }}
        
        slider_m.addEventListener("input", update);
        slider_p.addEventListener("input", update);

        const onload = () => {{
            const ctx = document.getElementById('chart_custom').getContext('2d');

            const data = {{
                labels: [min_x, max_x],
                datasets: [
                    {{
                        type: 'line',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        fill: false,
                        pointRadius: 0,  // Hides the points on the line
                    }},
                    {{
                        data: [{points_2_str}],
                        type: 'scatter',
                        borderColor: 'blue',
                        backgroundColor: 'blue',
                        pointRadius: 1,
                        pointHoverRadius: 3,
                        label: 'r = 2',
                        showLine: false,
                    }},
                    {{
                        data: [{points_7_str}],
                        type: 'scatter',
                        borderColor: 'orange',
                        backgroundColor: 'orange',
                        pointRadius: 1,
                        pointHoverRadius: 3,
                        label: 'r = 7',
                        showLine: false,
                    }},
                ]
            }};

            const config = {{
                data: data,
                options: {{
                    scales: {{
                        x: {{
                            type: 'linear',
                            //position: 'center',
                            min: min_x,   // Minimum value for x-axis
                            max: max_x   // Maximum value for x-axis
                        }},
                        y: {{
                            type: 'linear',
                            //position: 'center',
                            min: min_x,   // Minimum value for y-axis, adjust as necessary
                            max: max_x,   // Maximum value for y-axis, adjust as necessary
                        }}
                    }},
                }}
            }};

            chart = new Chart(ctx, config);
            update()
        }};

        window.mathadata.import_js_script("https://cdn.jsdelivr.net/npm/chart.js", onload)
    }}
    func()
    """

    run_js(js_code)

# TODO
# run_js("""
#     // Create a MutationObserver instance
#     const observer = new MutationObserver(function(mutations) {
#         console.log("working")
#         mutations.forEach(function(mutation) {
#             mutation.addedNodes.forEach(function(node) {
#                 console.log("node added")
#                 console.log(node)
#                 // Check if the added node is an element with the specified ID
#                 if (node.id === "container_chart") {
#                     console.log("setup charts")
#                     Jupyter.notebook.kernel.execute("setup_charts()")
#                 } else if (node.id === "container_chart_custom") {
#                     console.log("setup charts 2")
#                     Jupyter.notebook.kernel.execute("setup_charts_2()")
#                 }
#             });
#         });
#     });

#     // Start observing the document for mutations
#     observer.observe(document, { childList: true, subtree: true });
# """)

g_m = None
g_p = None
def compute_score(m, p):
    global g_m, g_p
    g_m = m
    g_p = p
    c_train = compute_c_train()
    error = erreur_lineaire(m, p, c_train)
    return error

def calculer_score_etape_2():
    global g_m, g_p
    if compute_score(g_m, g_p) > 8:
        print_error("Exécutez cette cellule quand vous aurez un score de moins de 8%.")
        return
    
    deux_caracteristiques = get_variable('deux_caracteristiques')
    def algorithme(d):
        k1, k2 = deux_caracteristiques(d)
        if k2 > g_m * k1 + g_p:
            return 2
        else:
            return 7
        
    def cb(score):
        validation_score_droite()

    calculer_score(algorithme, method="2 moyennes", parameters=f"m={g_m}, p={g_p}", cb=cb) 

def calculer_score_etape_3():
    global g_m, g_p 
    if compute_score(g_m, g_p) <= 50:
        above = 2
        below = 7
    else:
        above = 7
        below = 2
    
    deux_caracteristiques = get_variable('deux_caracteristiques')
    def algorithme(d):
        k1, k2 = deux_caracteristiques(d)
        if k2 > g_m * k1 + g_p:
            return above
        else:
            return below
        
    def cb(score):
        validation_score_droite_custom()

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"m={g_m}, p={g_p}", cb=cb) 
 
### Validation

def on_success_chart_1(answers):
    setup_charts()

def on_success_2_caracteristiques(answers):
    affichage_2_geo(display_k=True)

def on_succes_execution_caracteristiques_custom(answers):
    setup_charts_2()

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append("Les coordonnées doivent être écrites entre parenthèses séparés par une virgule. Exemple : (3, 5)")
        return False
    if len(coords) != 2:
        errors.append("Les coordonnées doivent être composées de deux valeurs séparés par une virgule. Pour les nombres à virgule, utilisez un point '.' et non une virgule")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not (isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float))):
        errors.append("Les coordonnées doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    return True

def function_validation_2_points(errors, answers):
    if not has_variable('deux_caracteristiques'):
        errors.append("Exécutez la cellule précédente pour définir la fonction deux_caracteristiques")
        return False
    
    A = answers['A']
    B = answers['B']
    if not check_coordinates(A, errors) or not check_coordinates(B, errors):
        return False
    
    deux_caracteristiques = get_variable('deux_caracteristiques')
    
    A_true = deux_caracteristiques(d_train[0])
    B_true = deux_caracteristiques(d_train[1])
    
    distA = np.sqrt((A[0] - A_true[0])**2 + (A[1] - A_true[1])**2)
    distB = np.sqrt((B[0] - B_true[0])**2 + (B[1] - B_true[1])**2)
    
    if distA > 3:
        distARev = np.sqrt((A[1] - A_true[0])**2 + (A[0] - A_true[1])**2)
        distAB = np.sqrt((A[0] - B_true[0])**2 + (A[1] - B_true[1])**2)
        if distAB < 3:
            errors.append("Les coordonnées de A ne sont pas correctes. Tu as peut être donné les coordonnées du point B à la place ?")
        elif distARev < 3:
            errors.append("Les coordonnées de A ne sont pas correctes. Attention, la première coordonnée est l'abscisse k1 et la deuxième l'ordonnée k2.")
        else:
            errors.append("Les coordonnées de A ne sont pas correctes.")
    if distB > 3:
        distBRev = np.sqrt((B[1] - B_true[0])**2 + (B[0] - B_true[1])**2)
        distAB = np.sqrt((B[0] - A_true[0])**2 + (B[1] - A_true[1])**2)
        if distAB < 3:
            errors.append("Les coordonnées de B ne sont pas correctes. Tu as peut être donné les coordonnées du point A à la place ?")
        elif distBRev < 3:
            errors.append("Les coordonnées de B ne sont pas correctes. Attention, la première coordonnée est l'abscisse k1 et la deuxième l'ordonnée k2.")
        else:
            errors.append("Les coordonnées de B ne sont pas correctes.")

def function_validation_equation(errors, answers):
    m = g_m
    p = g_p
    abscisse_M = answers['abscisse_M']
    ordonnee_M = answers['ordonnee_M']
    
    if not (isinstance(abscisse_M, (int, float)) and isinstance(ordonnee_M, (int, float))):
        errors.append("Les coordonnées de M doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    
    if abscisse_M != pointA[0]:
        errors.append("L'abscisse de M n'est pas correcte.")
        return False
    
    if ordonnee_M != m*abscisse_M + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        return False

    return True

validation_execution_2_caracteristiques = MathadataValidate(success="")
validation_question_2_caracteristiques = MathadataValidateVariables({
    'r1': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r1 n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    },
    'r2': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r2 n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    }
}, success="C'est la bonne réponse. L'image de 7 a presque la même moyenne sur la moitié haute et la moitié basse. L'image de 2 a une moyenne plus élevée sur la moitié basse car il y a plus de pixels blancs.",
    on_success=on_success_2_caracteristiques)

validation_execution_2_points = MathadataValidate(success="")
validation_question_2_points = MathadataValidateVariables({
    'A': None,
    'B': None,
}, function_validation=function_validation_2_points)
validation_execution_200_points = MathadataValidate(success="")
validation_question_couleur = MathadataValidateVariables({
    'classe_points_bleus': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_bleus n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    },
    'classe_points_oranges': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_oranges n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    }
})
validation_execution_10_points = MathadataValidate(success="")
validation_question_score_droite = MathadataValidateVariables({
    'score_10': {
        'value': 20,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 100,
                },
                'else': "Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100."
            },
            {
                'value': 2,
                'if': "Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreur et non le pourcentage d'erreur."
            }
        ]
    }
}, success="C'est la bonne réponse. Il y a un point bleu sous la droite et un point orange au dessus soit deux erreurs donc 20%.", on_success=on_success_chart_1)
validation_score_droite = MathadataValidate(success="")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_equation = MathadataValidateVariables({
    'abscisse_M': 20,
    'ordonnee_M': None
}, function_validation=function_validation_equation)
validation_execution_caracteristiques_custom = MathadataValidate(success="", on_success=on_succes_execution_caracteristiques_custom)
validation_score_droite_custom = MathadataValidate(success="")