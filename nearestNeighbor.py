import pickle
import numpy as np
import os

#this code was not originally in the Stanford class on how to process the data and the walk through I got from ChatGpt but I am like 95% sure it processes this right 
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

#this functions loads the data from the training sets and assigns it's output to the four variables 
#Xtr training images (50,000)
#Ytr training image labels (50,000)
#Xte test images (10,000)
#Yte test labels (10,000)
Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py') 
#print("done!")
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) / 255.0 # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) / 255.0 # Xte_rows becomes 10000 x 3072

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) 
      #I'm going to use L2 distance instead 

      distances = np.sqrt(np.sum((self.Xtr - X[i, :])**2, axis=1))
      #min_index = np.argmin(distances) # get the index with smallest distance
      #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

      #this allegedly will make the model run faster
      k = 5  # Use the 5 closest training points
      sorted_indices = np.argsort(distances)[:k]
      nearest_labels = self.ytr[sorted_indices]
      Ypred[i] = np.bincount(nearest_labels).argmax()  # Pick the most common label

    return Ypred

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)

accuracy = np.mean(Yte_predict == Yte)
print(f'accuracy: {accuracy:.6f}')

print("done!")