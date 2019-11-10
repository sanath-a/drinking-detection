import numpy as np
import cv2
import os
import csv
from skimage.feature import hog
from skimage.exposure import histogram
def import_data(train_set = 'original'):
    '''
    Loads data into X_train, X_test, X_val and y_train, y_test, y_val.
    All feature vectors are normalized by subtracting the mean pixel and dividing
    by the standard deviation.

    Assumes path folder includes folders named train test and val
    Assumes image size is (360, 640,3)

    ---Params---
    path: Path to image data set
    ------------
    ---Returns---
    X_train (nd.array): Training feature vectors of shape (N,360,640,3)
    y_train (nd.array): Training targets of shape (N,)
    X_test (nd.array): Test feature vectors of shape (M,360,640,3)
    y_test (nd.array): Test targets of shape (M,)
    X_val (nd.array): Validation feature vectors of shape (Z,360,640,3)
    y_val (nd.array): Validation targets of shape (Z,)
    -------------
    '''
    
    if train_set == 'original': 
        path = './data/original/'
    elif train_set == 'masked': 
        path = './data/masked/'
    else: 
        raise ValueError('train_set must be either original or masked')

    sets = ['train', 'test', 'val']
    X = []
    y = []
    with open('annotated.csv', mode = 'r') as file:
        reader = csv.DictReader(file, fieldnames = ('name', 'target'))
        target_dict = {}
        for row in reader:
            if row['target'] == 'yes':
                target_dict[row['name']] = 1
            else:
                target_dict[row['name']] = 0
    
    for s in sets:
        s_path = path + s + '/'
        test_img_names = [f for f in os.listdir(s_path)]
        test_img_paths = [s_path + f for f in test_img_names if os.path.splitext(s_path + f)[1] == '.jpg']
        N = len(test_img_paths)
        arr = np.zeros((N,360,640,3))
        targets = np.zeros((N,))
        for i in range(N):
            im = cv2.imread(test_img_paths[i])
            arr[i] = im
            targets[i] = target_dict[test_img_names[i]]
        X.append(arr)
        y.append(targets)


    X_train = X[0]
    X_test = X[1]
    X_val = X[2]

    y_train = y[0]
    y_test = y[1]
    y_val = y[2]

    ##Normalize Data: Subtracting mean pixel and dividing by std
    mean_pixel = X_train.mean(axis = (0,1,2), keepdims = True)
    std_pixel = X_train.std(axis = (0,1,2), keepdims = True)

    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel

    return X_train, y_train, X_test, y_test, X_val, y_val

def Feature_Extraction(X):
    hist_X = im_hist(X)
    hog_X = im_hog(X)

    features = np.concatenate((hist_X, hog_X), axis = 1)
    return features

def im_hist(X):
    N = X.shape[0]
    hist_array = np.zeros((N,256 * 3))
    for i in range(N):
        h_list = []
        for j in range(3):
            v = X[i,:,:,j]
            hist, bin_centers = histogram(v)
            h_list.append(hist)
        vector = np.array(h_list).ravel()
        hist_array[i] = vector
    return hist_array

def im_hog (X):
    N = X.shape[0]
    feature_array = np.zeros((N,271674))
    for i in range(N):
        im = X[i]
        hog_vec = hog(im, multichannel = True)
        feature_array[i] = hog_vec
    return feature_array


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))
if __name__ == '__main__':
    print ('Loading all datasets..')
    X_train, y_train, X_test, y_test, X_val, y_val = import_data()
    X_train_feat = Feature_Extraction(X_train)
    print (X_train_feat.shape)
    print (X_train.shape, X_test.shape, X_val.shape)
