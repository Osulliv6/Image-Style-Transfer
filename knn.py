import pickle
import numpy as np
import os

#I am loading the data in the same way that I did in the nn program since it is the same data 
def load_CIFAR10(data_dir):
    def batchLoad(file):
        with open(file, 'rb') as fo: #opens binary file to read 
            batch = pickle.load(fo, encoding='bytes')
            #pickling is a python process that converts data structures to binary or binary back into data structures 
            #so for example this code reads the binary data and converts it into a dictionary 
            #encoding='bytes' is included because the dataset was originally written in Python2 
        return batch[b'data'], np.array(batch[b'labels'])
        #b'data is the individual pixel values 
        #batch[b'data] stores the different images 
        #Key b'labels' stores class labels.
        #batch[b'labels'] is a list of 10,000 integers (values 0 to 9).
    
    X_train, y_train = [], []
    #X_train stores the image pixels for the training set 
    #Y_train stores the image labels for the training set 

    for i in range(1, 6): #1-6 because there are 5 training batches of 10,000 images
        data, labels = batchLoad(os.path.join(data_dir, f"data_batch_{i}"))
        #extracts the image pixel values, shape, and the class labels 
        X_train.append(data)
        y_train.append(labels)
        #this data is then stored in xtrain and ytrain as lists 
    
    #right now the arrays are 5 arrays with the five different bathces this is just combining them 
    X_train = np.concatenate(X_train)  
    y_train = np.concatenate(y_train)

    # Load test data (10,000 images)
    X_test, y_test = batchLoad(os.path.join(data_dir, "test_batch"))

    # Reshape the images from (N, 3072) → (N, 32, 32, 3)
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return X_train, y_train, X_test, y_test


Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')

# Flatten images for KNN
Xtr_rows = Xtr.reshape(Xtr.shape[0], -1)
Xte_rows = Xte.reshape(Xte.shape[0], -1)

class KNNClassifier:
    def __init__(self, k=5, batch_size=500):
        self.k = k
        self.Xtr = None
        self.ytr = None
        self.batch_size = batch_size  # Initialize batch_size

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):  # Process one test sample at a time
            distances = np.sqrt(np.sum((self.Xtr - X[i, :]) ** 2, axis=1))  # Vectorized L2 distance
            k_neighbors = np.argpartition(distances, self.k)[:self.k]  # Get k nearest neighbors
            k_labels = self.ytr[k_neighbors]  # Get their labels
            Ypred[i] = np.bincount(k_labels).argmax()  # Assign the most common label

        return Ypred
# Instantiate KNN with k=5
for k in [1, 3, 5, 10, 20]:
    knn = KNNClassifier(k=k)
    knn.train(Xtr_rows, Ytr)
    Yte_predict = knn.predict(Xte_rows)

    # Compute accuracy
    accuracy = np.mean(Yte_predict == Yte)
    print(f'Accuracy: {accuracy:.6f}')