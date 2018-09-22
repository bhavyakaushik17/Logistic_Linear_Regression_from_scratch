import numpy as np
import pandas as pd
import sys

train = pd.read_csv(sys.argv[2],header = None).iloc[:, :]
test = pd.read_csv(sys.argv[3],header = None).iloc[:, :]

X = train.iloc[:, 1:]
## adding one more dummy feature to support the bias term (b = w0)
X.loc[:,-1] = np.ones(len(train))
X = np.array(X)

y = train.iloc[:, 0]
y = np.array(y)

X_test = test.iloc[:, 1:]
X_test.loc[:,-1] = np.ones(len(test))
X_test = np.array(X_test)

y_test = test.iloc[:, 0]
y_test = np.array(y_test)
y_test = np.array(y_test)

## NMSE helper function (given)
def compute_error(true_vals, predicted_vals):
    '''
        Compute normalized RMSE
        Args:
            true_vals: numpy array of targets
            predicted_vals: numpy array of predicted values
    '''
    # Subtract minimum value
    min_value = np.min(true_vals)
    error = np.sum(np.square(true_vals-predicted_vals))/np.sum(np.square(true_vals-min_value))
    return error

if (sys.argv[1] == 'a'):
    xTx = np.matmul(np.transpose(X),X)
    XtX = np.linalg.inv(xTx)
    XtX_xT = np.matmul(XtX,np.transpose(X))
    theta = np.matmul(XtX_xT,y)
    y_pred = np.matmul(theta, np.transpose(X_test))
    f = open(sys.argv[4], "a")
    for i in range(len(y_pred)):
        f.write(str(y_pred[i])+'\n')
    f.close()

if (sys.argv[1] == 'b'):
    X_split = np.array_split(X,10)
    y_split = np.array_split(y,10)
    ### Iterations for finding optimal regularization parameter
    l = [0.0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
    nmse = np.zeros(len(l))
    for i in range(len(l)):
        lamb = l[i]
        for j in range(10):
            ###
            temp = list(range(10))
            temp.remove(j)
            Xi_test = X_split[j]
            Xi = np.concatenate([X_split[k] for k in temp])
            yi_test = y_split[j]
            yi = np.concatenate([y_split[k] for k in temp])
            ###
            xTx = np.matmul(np.transpose(Xi),Xi)
            lambda_I = lamb * np.eye(xTx.shape[0])
            XtX_lambda = np.linalg.inv(xTx+lambda_I)
            XtX_lamb_xT = np.matmul(XtX_lambda,np.transpose(Xi))
            XtX_lamb_xT = np.matmul(XtX_lambda,np.transpose(Xi))
            theta = np.matmul(XtX_lamb_xT,yi)
            ###
            yi_pred = np.matmul(theta, np.transpose(Xi_test))
            nmse[i]+=compute_error(yi_test,yi_pred)
    ###
    nmse = nmse/10
    lamb = l[np.argmin(nmse)]
    xTx = np.matmul(np.transpose(X),X)
    lambda_I = lamb * np.eye(xTx.shape[0])
    XtX_lambda = np.linalg.inv(xTx+lambda_I)
    XtX_lamb_xT = np.matmul(XtX_lambda,np.transpose(X))
    XtX_lamb_xT = np.matmul(XtX_lambda,np.transpose(X))
    theta = np.matmul(XtX_lamb_xT,y)
    ####
    y_pred = np.matmul(theta, np.transpose(X_test))
    f = open(sys.argv[4], "a")
    for i in range(len(y_pred)):
        f.write(str(y_pred[i])+'\n')
    f.close()

if (sys.argv[1] == 'c'):
    X_split = np.array_split(X,10)
    y_split = np.array_split(y,10)
    from sklearn.linear_model import LassoLars
    l = [0.0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
    nmse = np.zeros(len(l))
    for i in range(len(l)):
        alpha = l[i]
        regressor = LassoLars(alpha)
        for j in range(10):
            ###
            temp = list(range(10))
            temp.remove(j)
            Xi_test = X_split[j]
            Xi = np.concatenate([X_split[k] for k in temp])
            yi_test = y_split[j]
            yi = np.concatenate([y_split[k] for k in temp])
            ###
            regressor.fit(Xi, yi)
            ###
            yi_pred = regressor.predict(Xi_test)
            nmse[i]+=compute_error(yi_test,yi_pred)
    ###
    nmse = nmse/10
    ###
    alpha = l[np.argmin(nmse)]
    regressor = LassoLars(alpha)
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    f = open(sys.argv[4], "a")
    for i in range(len(y_pred)):
        f.write(str(y_pred[i])+'\n')
    f.close()