import numpy as np
import cv2
import os
import csv
def import_data(path = './data/original/'):
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


if __name__ == '__main__':
    print ('Loading all datasets..')
    X_train, y_train, X_test, y_test, X_val, y_val = import_data()
    print (X_train.shape, X_test.shape, X_val.shape)
