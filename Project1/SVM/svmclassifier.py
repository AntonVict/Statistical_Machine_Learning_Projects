import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()


#choose a seed
seed = 42
np.random.seed(seed)
# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape
Ntest = int(100)
Ntrain = int(1697)
Xtrain = X[0:Ntrain,:]
ytrain = y[0:Ntrain]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]


#should we even compute the hinge-loss??
#oohhhh this one should also return the states
def multiClassHingeLoss(Theta, x, y):
    margins = [] 
    scores = []
    numOfClasses = Theta.shape[1]
    for j in range(numOfClasses):
        # if j != y:
        score = np.dot(x, (Theta[:,j]- Theta[:,y]))
        margins.append(1 + score)
            # scores.append(score)
            
    # margins.append()
    return np.max(margins), margins


def svmsubgradient(Theta, x, y):
#  Returns a subgradient of the objective empirical hinge loss
#
# The inputs are Theta, of size n-by-K, where K is the number of classes,
# x of size n, and y an integer in {0, 1, ..., 9}.
    G = np.zeros(Theta.shape)
    
    hingeLoss, margins = multiClassHingeLoss(Theta, x, y)
    numOfClasses = Theta.shape[1]
    if hingeLoss != 0:
        for j in range(numOfClasses):
            if j != y:
                if margins[j] > 0:
                    G[:, j] += x
                    G[:, y] -= x
    return G, hingeLoss

def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000):
#
# Performs maxiter iterations of projected stochastic gradient descent
# on the data contained in the matrix Xtrain, of size n-by-d, where n
# is the sample size and d is the dimension, and the label vector
# ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
# classification matrices Theta and mean_Theta, where the first is the final
# point of SGD and the second is the mean of all the iterates of SGD.
#
# Each iteration consists of choosing a random index from n and the
# associated data point in X, taking a subgradient step for the
# multiclass SVM objective, and projecting onto the Euclidean ball
# The stepsize is init_stepsize / sqrt(iteration).
    K = 10
    NN, dd = Xtrain.shape
    print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K
    ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT
    
    stepsize = init_stepsize/math.sqrt(K)
    size_Dataset = Xtrain.shape[0]
    loss_history = []
    for _ in range(maxiter):
        random_sample = random.randint(0, size_Dataset-1)
        subGradient, loss = svmsubgradient(Theta, Xtrain[random_sample], ytrain[random_sample])
        Theta -= stepsize*subGradient

        frobenius_norm = np.linalg.norm(Theta, ord='fro')
        if frobenius_norm >= l2_radius*2:
            Theta = (l2_radius/frobenius_norm)*Theta
        
        loss_history.append(loss)
        # projection set each value in theta to min(theta_ij, l2_radius)
        # apparently the frobenius norm can be computed like this
    
    return Theta, mean_Theta, loss_history

def Classify(Xdata, Theta):
#
# Takes in an N-by-d data matrix Adata, where d is the dimension and N
# is the sample size, and a classifier X, which is of size d-by-K,
# where K is the number of classes.
#
# Returns a vector of length N consisting of the predicted digits in
# the classes.
    scores = np.matmul(Xdata, Theta)
    inds = np.argmax(scores, axis = 1)
    return(inds)



l2_radius = 40.0
M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
init_stepsize = l2_radius/M_raw
maxiter = 40000
Theta, mean_Theta, loss_history = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius)

# Plot the loss history
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.show()

num_classes = Theta.shape[1]

plt.figure(figsize=(10, 5))
for i in range(num_classes):
    plt.subplot(2, 5, i + 1)
    plt.imshow(Theta[:, i].reshape(8, 8), cmap='gray')
    plt.title(f' {i}')
    plt.axis('off')

plt.show()

print('Error rate')
print(np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest))

