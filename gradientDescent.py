import numpy as np
import matplotlib.pyplot as plt  


def h(Xi,theta):
    return Xi.dot(theta)

def cost(X,y,theta):
    predicted=[]
    for i in range(X.shape[0]):
        predicted.append(h(X[i],theta))
    predicted=np.array(predicted)
    print("predicted:")
    print(predicted[:2])
    return ((1/2)*np.sum(np.square(predicted-y)))



#load data
file=open("./3D_spatial_network.txt","rt")
X=[] #input features, each row refers to separate training example- X[0] refers to 0th training example
y=[] #target attribute, each row refers to separate training example- y[0] refers to 0th training example
for row in file:
    X_curr=[]
    X_curr.append(1.0)
    y_curr=[]
    values=row.split(sep=',')
    X_curr.append(float(values[1]))
    X_curr.append(float(values[2]))
    y_curr.append(float(values[3]))
    X.append(X_curr)
    y.append(y_curr)
X=np.array(X)
y=np.array(y)
print(X[:2])
print(y[:2])


#theta = weights
theta=np.ones((3,1)) #[[0],[0],[0]]
print(theta)
#print(theta.transpose())  #[0,0,0]
print(h(X[0],theta))
print(cost(X,y,theta))







        



    


