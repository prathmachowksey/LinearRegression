import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from mpl_toolkits import mplot3d

def predicted(X,theta):
    return (X@(np.transpose(theta)))

def cost(X,y,theta):
     return ((1/(2*len(X)))*np.sum(np.square((predicted(X,theta))-y)))
     #return ((1/2)*np.sum(np.square((predicted(X,theta))-y)))

def gradient_descent(X,y,theta,learning_rate,iterations,precision):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        print("iteration ",i)
        theta = theta - (learning_rate/len(X)) * np.sum(X * ((predicted(X,theta)) - y), axis=0)
        #theta = theta - ((learning_rate) * np.sum(X * ((predicted(X,theta)) - y), axis=0))
        print(theta)
        cost_history[i] = cost(X, y, theta)
        #if(i>0 and (cost_history[i]-cost_history[i-1])<precision):
            #break
    
    return theta,cost_history
def r_squared(X,y,final_weights):

    return ((np.sum(np.square(predicted(X,final_weights)-y.mean())))/(np.sum(np.square(y-y.mean()))))

def rmse(X,y,final_weights):
    return (np.sqrt(np.square(predicted(X,final_weights)-y).mean()))

def rmse_2(X,y,final_weights):
    return math.sqrt(mean_squared_error(y,predicted(X,final_weights)))

def squared_error(X,y,final_weights):
    return np.sum(np.square(predicted(X,final_weights)-y))
    
#load data
data=pd.read_csv('3D_spatial_network.txt',names=["id","latitude","longitude","altitude"])

#drop the first column
data=data.drop("id",axis=1)

#normalise data
data=(data-data.mean())/data.std()


#create X and y matrices
X=data.iloc[:,0:2]
X=X.values
y=data.iloc[:,2:3]
y=y.values

#normalise X
'''
mean = np.ones(X.shape[1])
std = np.ones(X.shape[1])
for i in range(0, X.shape[1]):
    mean[i] = np.mean(X.transpose()[i])
    std[i] = np.std(X.transpose()[i])
    for j in range(0, X.shape[0]):
        X[j][i] = (X[j][i] - mean[i])/std[i]
'''

#normalise y
#y=(y-y.mean())/y.std()

#sklearn for reference:
print("sklearn w0 w1 w2")
reg=LinearRegression().fit(X,y)

print(reg.intercept_)
print(reg.coef_)
#concatenate ones in X
ones=np.ones((X.shape[0],1))
X=np.concatenate((ones,X),axis=1)
#weights
theta = np.ones((1,3))


#set  parameters
learning_rate = 0.00007
iterations = 1000
precision=0.000001
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#using sklearn:
reg=LinearRegression(X_train[:][1:])

final_weights,cost_history = gradient_descent(X_train,y_train,theta,learning_rate,iterations,precision)
print("Weights: "+str(final_weights))

final_cost = cost(X,y,final_weights)
print("final cost:"+str(final_cost))

r2_train=r_squared(X_train,y_train,final_weights)
print("r2 for training dataset: "+str(r2_train))
r2_test=r_squared(X_test,y_test,final_weights)
print("r2 for testing dataset: "+str(r2_test))

rmse_train=rmse(X_train,y_train,final_weights)
print("rmse for training dataset: "+str(rmse_train))
rmse_test=rmse(X_test,y_test,final_weights)
print("rmse for testing dataset: "+str(rmse_test))

rmse_train=rmse_2(X_train,y_train,final_weights)
print("rmse-sklearn for training dataset: "+str(rmse_train))
rmse_test=rmse_2(X_test,y_test,final_weights)
print("rmse-sklearn for testing dataset: "+str(rmse_test))


squared_error_train=squared_error(X_train,y_train,final_weights)
print("squared_error for training dataset: "+str(squared_error_train))
squared_error_test=squared_error(X_test,y_test,final_weights)
print("squared_error for testing dataset: "+str(squared_error_test))

plt.plot(np.arange(iterations),cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()


