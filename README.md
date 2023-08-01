# svmClassifier
An svm image classifier for the popular MNIST dataset,
utilizing PCA for dimension reduction (from sklearn), standard scaling  and the SVM module, again from sklearn.

The aim of the project is to become familiar with Support Vector Machines and the 
their practical implementation. This project is about a classifier with svm, i.e. 
Support Vector Classifier, to separate odd/even numbers from the known Mnist image dataset.
The data are loaded using PyTorch, or rather its ready datasets and with "target_transform = Lambda(lambda y: 1 if(y % 2) else -1)", the labels are seperated into our two classes.

The 28 by 28 images (torches) are then flattened (784 x 1) numpy arrays. An important aspect of this project was the use of PCA for dimensionality reduction, as the training of the SVM models, using the whole input was counterproductive. 
First, the entire dataset had to be centered and scaled (using scikit's StandardScaler), before begining the PCA proccess.
After PCA, which was performed once for the whole data set,  the input was reduced to 245 pricipal components - features, to be used for training and testing. The number 245 was chosen after examining the scree-plot, and contains roughly 90% of the original information, using only a third of the size. 

The kernels tested where the linear kernel (LinearSVC), polynomial kernel and rbf kernel. 
There were several tests conducted on each kernel, for various coeffiecient, degrees and training iterations.
The best polynomial kernel was that of 5th degree with coef = 1, achieving a 98,6% testing accuracy, with 1000 training iterations,
while the best rbf kernel reached a 97,6% testing accuracy with 2000 iterations.

These results were compared with the result of KNN classification and nearest centroid classification, (98% and 80% respectively).
