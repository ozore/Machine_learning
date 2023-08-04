import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Définition du dataset 
x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

# Initialisation des paramètres du modèle 
w = 200
b = 100

# Initialisation du modèle - Fw,b(x)


def model(x_i, w_p, b_p): # non utilisé dans le calcul de la cost function
    retour = x_i*w_p + b_p
    # print(f"{retour} = {x_i} * {w_p} + {b_p} ")
    return retour


# Initialisation de la COST function 
""" 
x : features 
y : target
w, b :paramètres
m : taille du dataset
"""


def compute_cost(x, y, w, b):
    m = x.shape[0] # Taille du dataset
    sum = 0

    for i in range(m):
        f_wb = x[i]*w +b
        sum = sum + (f_wb - y[i])**2
    total_cost = (1/2*m) * sum
    return total_cost



# Initiliasaiton de la fonction de compute Gradient (calcul des dérivées) 


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0 
    dj_db =0 

    for i in range(m):
        f_wb = x[i]*w +b
        dj_dw_i = (f_wb -y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw =+ dj_dw_i
        dj_db =+ dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(w_in, b_in, x, y, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # calcul les gradients 
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Calcul des parametres
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        # Sauvegarde de Cost à chaque itération
        J_history.append(cost_function(x, y, w, b))
        p_history.append([w,b])

        # Afficher la valeur de COST tous les 10 intervales 
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history

# MAIN 
alpha = 1.0e-2
iterations = 10000
w_init = 0
b_init = 0 


w_final, b_final, J_hist, p_hist = gradient_descent(w_init, b_init, x_train, y_train, alpha ,iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# compute_cost(x_train, y_train, w, b)
# compute_gradient(x_train, y_train, w, b)

