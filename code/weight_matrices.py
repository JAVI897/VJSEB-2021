from sklearn.metrics import pairwise_distances
import numpy as np

def compute_W_simple_kernel(X, Y, eps):
    # Simple kernel
    n = X.shape[0]
    dist_matrix = pairwise_distances(X)
    nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < eps and index != i and Y[index] == Y[i]] for i in range(n) ])
    # Weight matrix
    W = []
    for i in range(n):
        w_aux = np.zeros((1, n))
        similarities = np.array([ 1 for v in nn_matrix[i]] )
        np.put(w_aux, nn_matrix[i], similarities)
        W.append(w_aux[0])
    W = np.array(W)
    return W

def compute_W_multiquadric_kernel(X, Y, eps, c):
    
    def multiquadric(x, y, c):
        """
        k(x, y) = sqrt(||x-y||^2 + c^2)
        Hiperparámetros: c
        """
        return np.sqrt(np.linalg.norm(x-y)**2 + c**2)
    
    n = X.shape[0]
    dist_matrix = pairwise_distances(X)
    nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < eps and index != i and Y[index] == Y[i]] for i in range(n) ])
    # Weight matrix
    W = []
    for i in range(n):
        w_aux = np.zeros((1, n))
        similarities = []
        for v in nn_matrix[i]:
            similarities.append(multiquadric(X[i,:], X[v,:], c))
        similarities = np.array(similarities)
        np.put(w_aux, nn_matrix[i], similarities)
        W.append(w_aux[0])
    W = np.array(W)
    return W