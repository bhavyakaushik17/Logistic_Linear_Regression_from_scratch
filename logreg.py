import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
import sys

train = pd.read_csv(sys.argv[6],header = None).iloc[:, :]
test = pd.read_csv(sys.argv[8],header = None).iloc[:, :]

X = train.iloc[:, 1]
X_test = test.iloc[:, 1]

y = train.iloc[:, 0]
y = np.array(y)
y_test = test.iloc[:, 0]
y_test = np.array(y_test)

vocab = []
with open(sys.argv[7], encoding = 'utf8') as vocabfile:
    for line in vocabfile:
        vocab.append(line.split('\n')[0])
## Preprocessing
rmlist = ['.', '?', '!', ',', ';', ':', '{', '}', '[', ']', '(', ')', '+', '/', '*', '=', '>', '<', '"']
def preprocess(X):
    for char in rmlist:
        X = X.replace(char,' ')
    X = X.lower()
    return X

for i in range(len(X)):
    X.at[i] = preprocess(X[i])
for i in range(len(X_test)):
    X_test.at[i] = preprocess(X_test[i])

################################################################################################
## Creating Sparse Feature Matrices
#####
indptr = [0]
indices = []
data = []
for i in range(len(X)):
    for word in X[i].split():
        if word in vocab:
            index = vocab.index(word)
            indices.append(index)
            data.append(1)
    indptr.append(len(indices))
#####
indices.append(len(vocab)-1)
data.append(0)
indptr.pop()
indptr.append(len(indices))
x = csr_matrix((data, indices, indptr), dtype=int)
x = hstack([x,csr_matrix(np.ones((len(X),1)))],format='csr')
#####
indptr = [0]
indices = []
data = []
for i in range(len(X_test)):
    for word in X_test[i].split():
        if word in vocab:
            index = vocab.index(word)
            indices.append(index)
            data.append(1)
    indptr.append(len(indices))
#####
indices.append(len(vocab)-1)
data.append(0)
indptr.pop()
indptr.append(len(indices))
x_test = csr_matrix((data, indices, indptr), dtype=int)
x_test = hstack([x_test,csr_matrix(np.ones((len(X_test),1)))],format='csr')
################################################################################################
## Logistic Regression Classes

def compute_accuracy(true_labels, predicted_labels):
    num_instances = true_labels.size
    return np.sum(true_labels==predicted_labels)/num_instances

class LogisticRegression_fullbatch:

    def __init__(self, algo = 3, lr=0.05, num_iter=500, lamb=0):
        self.lr = lr
        self.num_iter = num_iter*2 # for better accuracy (doesn't take much time)
        self.lamb = lamb
        self.algo = algo

    def __sigmoid(self, z):
        return np.ones(z.shape) / (np.ones(z.shape) + np.exp(-z))

    def __loss(self, h, y, theta):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (self.lamb/2)*(theta.dot(theta))/y.shape[0]

    def fit(self, x, y):

        # initializing weights
        self.theta = 0.001*(np.random.rand(x.shape[1]))

        for i in range(self.num_iter):
            z = x.dot(self.theta)
            h = self.__sigmoid(z)
            gradient = (x.transpose().dot((h - y)) - self.lamb*self.theta)/y.shape[0]

            if (self.algo == 1):
              self.theta -= self.lr * gradient

            if (self.algo == 2):
              lri = self.lr/((i+1)**0.5)
              self.theta -= lri * gradient

            if (self.algo == 3):
              lr1 = 0
              lr2 = self.lr
              mid = (lr1+lr2)/2
              n_stuck = 0
              theta1 = self.theta
              fdash = 0
              while(n_stuck<2):
                theta1 -= mid * gradient
                z1 = x.dot(theta1)
                h1 = self.__sigmoid(z1)
                fdash = gradient.T.dot(x.transpose().dot((h1 - h)))
                if (fdash>0):
                  lr2 = mid
                else:
                  lr1 = mid
                mid = (lr1+lr2)/2
                n_stuck+=1
              self.theta -= mid * gradient
              ####

    def predict_prob(self, x):
        return self.__sigmoid(x.dot(self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold



class LogisticRegression_minibatch:

    def __init__(self, algo = 1, lr=0.05, num_iter=500, lamb=0, batch_size = 128):
        self.lr = lr/5 # as higher number of epochs
        self.num_iter = num_iter//5 # for decreasing time complexity
        self.lamb = lamb
        self.algo = algo
        self.batch_size = batch_size

    def __sigmoid(self, z):
        return np.ones(z.shape) / (np.ones(z.shape) + np.exp(-z))

    def __loss(self, h, y, theta):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (self.lamb/2)*(theta.dot(theta))/y.shape[0]

    def fit(self, x, y):

        # initializing weights
        self.theta = 0.001*(np.random.rand(x.shape[1]))

        indlist = list(range(0,x.shape[0],self.batch_size))
        indlist.append(x.shape[0]-1)
        x_split = [x[indlist[i]:indlist[i+1]] for i in range(len(indlist)-1)]
        y_split = [y[indlist[i]:indlist[i+1]] for i in range(len(indlist)-1)]

        for i in range(self.num_iter):
            for j in range(len(y_split)):

                z = x_split[j].dot(self.theta)
                h = self.__sigmoid(z)
                gradient = (x_split[j].transpose().dot((h - y_split[j])) - self.lamb*self.theta)/y_split[j].shape[0]

                if (self.algo == 1):
                  self.theta -= self.lr * gradient

                if (self.algo == 2):
                  lri = self.lr/((i+1)**0.5)
                  self.theta -= lri * gradient

                if (self.algo == 3):
                  lr1 = 0
                  lr2 = self.lr
                  mid = (lr1+lr2)/2
                  n_stuck = 0
                  theta1 = self.theta
                  fdash = 0
                  while(n_stuck<2):
                    theta1 -= mid * gradient
                    z1 = x_split[j].dot(theta1)
                    h1 = self.__sigmoid(z1)
                    fdash = gradient.T.dot(x_split[j].transpose().dot((h1 - h)))
                    if (fdash>0):
                      lr2 = mid
                    else:
                      lr1 = mid
                    mid = (lr1+lr2)/2
                    n_stuck+=1
                  self.theta -= mid * gradient
                  ####


    def predict_prob(self, x):
        return self.__sigmoid(x.dot(self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
################################################################################################
## MAIN
if (sys.argv[1] == 'a'):
        lamblist = [0.0,10,500]
        x_split = [x[i:i+x.shape[0]//10] for i in range(0,x.shape[0],x.shape[0]//10)]
        y_split = [y[i:i+y.shape[0]//10] for i in range(0,y.shape[0],y.shape[0]//10)]
        acc = np.zeros(len(lamblist))
        for i in range(len(lamblist)):
            lamb = lamblist[i]
            model = LogisticRegression_fullbatch(algo = int(sys.argv[2]), lr=float(sys.argv[3]), num_iter=int(sys.argv[4]), lamb=lamb)
            for j in range(10):
                ###
                temp = list(range(10))
                temp.remove(j)
                xi_test = x_split[j]
                xi = vstack([x_split[k] for k in temp],format='csr')
                yi_test = y_split[j]
                yi = np.concatenate([y_split[k] for k in temp])
                ###
                model.fit(xi,yi)
                ###
                predictedi = model.predict(xi_test,0.5)
                acc[i]+=compute_accuracy(yi_test, predictedi)
        acc = acc/10
        lamb  = lamblist[np.argmax(acc)]
        model = LogisticRegression_fullbatch(algo = int(sys.argv[2]), lr=float(sys.argv[3]), num_iter=int(sys.argv[4]), lamb=lamb)
        model.fit(x,y)
        predicted = model.predict(x_test,0.5)
        f = open(sys.argv[9], "a")
        for i in range(len(predicted)):
            f.write(str(int(predicted[i]))+'\n')    
        f.close()

############################################
if (sys.argv[1] == 'b'):
        lamblist = [0.0,10,500]
        x_split = [x[i:i+x.shape[0]//10] for i in range(0,x.shape[0],x.shape[0]//10)]
        y_split = [y[i:i+y.shape[0]//10] for i in range(0,y.shape[0],y.shape[0]//10)]
        acc = np.zeros(len(lamblist))
        for i in range(len(lamblist)):
            lamb = lamblist[i]
            model = LogisticRegression_minibatch(algo = int(sys.argv[2]), lr=float(sys.argv[3]), num_iter=int(sys.argv[4]), batch_size = int(sys.argv[5]), lamb=lamb)
            for j in range(10):
                ###
                temp = list(range(10))
                temp.remove(j)
                xi_test = x_split[j]
                xi = vstack([x_split[k] for k in temp],format='csr')
                yi_test = y_split[j]
                yi = np.concatenate([y_split[k] for k in temp])
                ###
                model.fit(xi,yi)
                ###
                predictedi = model.predict(xi_test,0.5)
                acc[i]+=compute_accuracy(yi_test, predictedi)
        acc = acc/10
        lamb  = lamblist[np.argmax(acc)]
        model = LogisticRegression_minibatch(algo = int(sys.argv[2]), lr=float(sys.argv[3]), num_iter=int(sys.argv[4]), batch_size = int(sys.argv[5]), lamb=lamb)
        model.fit(x,y)
        predicted = model.predict(x_test,0.5)
        f = open(sys.argv[9], "a")
        for i in range(len(predicted)):
            f.write(str(int(predicted[i]))+'\n')
        f.close()