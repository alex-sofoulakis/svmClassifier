# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:44:43 2021

@author: Alexandros
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import time

def neighbors(X,Y,Xtest,Ytest,neighbors): #nearest neighbors algorithm
    print("Starting neighbors")
    classifier = KNeighborsClassifier(n_neighbors=neighbors) 
    classifier.fit(X, Y) #train the classifier
    ycalc = classifier.predict(Xtest) #make test predictions
    print(classification_report(Ytest, ycalc)) #score the classifier

def nearestCentroid(X,Y,Xtest,Ytest): #nearest centoid classifier
    print("starting Centroid")
    classifier = NearestCentroid()
    classifier.fit(X,Y)
    ycalc = classifier.predict(Xtest)
    print(classification_report(Ytest, ycalc))


training_data = datasets.MNIST( #load data
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform = Lambda(lambda y: 1 if(y % 2) else -1) #format labels to 1/-1 if even/odd
)

test_data = datasets.MNIST( #load data
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform = Lambda(lambda y: 1 if(y % 2) else -1)
)


count = 0
shape = len(training_data),28*28
shape2 = len(test_data),28*28

ftrainX = np.empty(shape)
ftrainY = np.empty(len(training_data))

ftestX = np.empty(shape2)
ftestY = np.empty(len(test_data))

for x,y in training_data:
    new_x = torch.flatten(x,0,-1) #from 28*28 to 784
    new_x = new_x.numpy() #convert to numpy
    
    
    ftrainX[count] = new_x #transfer data to new arrays
    ftrainY[count] = y
    count+=1

count = 0
for x,y in test_data:
    new_x = torch.flatten(x,0,-1)
    new_x = new_x.numpy()
    
    ftestX[count] = new_x
    ftestY[count] = y
    count+=1

allData = np.concatenate((ftrainX,ftestX),0) #join the training data with the test data

scaler = StandardScaler().fit(allData) #scale and center all our data
allData = scaler.transform(allData)


#Data is now modifyed, centered and scaled and ready to be used
print("Data ready")


print(f"Original dimensions: {allData.shape}")
pca = PCA(n_components=245) #Perform Principal Components Analysis
pca.fit(allData)
finalX = pca.transform(allData)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
print(f"Percentage of information kept: {sum(per_var)}")
labels = [str(x) for x in range(1, len(per_var) +1)]

plt.bar(x = range(1,len(per_var) +1), height=per_var, tick_label= labels)
plt.show()

print(finalX.shape) #to see the reduced dimensions

Xtrain = finalX[:60000] #seperate training from test data
Xtest = finalX[60000:]

print(Xtrain.shape)
print(Xtest.shape)


startT = time.time()

#The various svms, comment out those not to be used at your discretion
#my_svm = svm.SVC(kernel='poly', degree=5, cache_size=1000, coef0=1) 
my_svm = svm.SVC(kernel='rbf', gamma=0.01, cache_size=1000, max_iter=2000)
#my_svm = svm.LinearSVC(C=0.01)
my_svm.fit(Xtrain,ftrainY)
print(f"Accuracy of training: {my_svm.score(Xtrain, ftrainY)}")

plot_confusion_matrix(my_svm, Xtest, ftestY, values_format = "d", display_labels = ['Artios', 'zygos']) #plotting confusion matrix for visualization of results
print(f"Accuracy of testing: {my_svm.score(Xtest, ftestY)}")
print(my_svm) 


neighbors(Xtrain, ftrainY, Xtest, ftestY,neighbors=3) #Knn call, change number of neighbors as you like
nearestCentroid(Xtrain, ftrainY, Xtest, ftestY)
endT = time.time()
print(f'Done! Total time for training and testing was {endT-startT}.')

    