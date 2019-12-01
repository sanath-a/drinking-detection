# Drinking Detection for HABits Lab

This project is meant for the HABits lab at Northwestern University.

# Overview
This project aims to use CNNs to detect when an actor wearing a camera is drinking. Actual videos are not included in this repository - only the scripts and models used to analyze them.

# Using this Repository
#### Dependencies
OpenCV2 is used for frame capturing.
Tensorflow is used for Neural Networks.
Numpy and Pandas are used for data wrangling.
These dependencies are listed in `requirements.txt`

#### Source Code
Within the `src` folder, you'll find the following files:

##### Model.py
`model.py` contains the class `ThreeLayerConvNet` used for the computer vision task It has a fairly simple architecture: Two convolutional layers and one fully connected layer. The model is imported in the Jupyter notebook for training.. The model inherits from the Keras model API. Useful functions:

`ThreeLayerConvNet.compile(optimizer, loss, metrics)`. This compiles the model. The Jupyter notebooks pass the result of the `optimizer_init_fn` as the optimizer (see below). The model must be compiled before training or testing.

`ThreeLayerConvNet.fit(X_train,y_train, batch_size, epochs, validation_data)`. Fits training data. See Keras documentation for more (link below).

`ThreeLayerConvNet.predict(X_test)` returns scores given test feature vectors.

`ThreeLayerConvNet.evaluate(X_val, y_val)` returns the loss and accuracy when evalauating on the given data in the form of the tuple `(loss, acc)`.

For more detail, check out the [Keras Documentation](https://keras.io/models/model/).

##### Optimizer.py
`optimizer.py` contains the `optimizer_init_fn`. This function has one argument: the learning rate. It returns a Keras optimizer to be passed to `ThreeLayerConvNet` during training.

##### Framecapture.py
`Framecapture.py` includes two functions (`extractImages` and `setBuilder` are helpers) to extract images from videos.

`Extractor(file_names, out_path = './data/', verbose = True)` extracts frames from a list of paths and randomly divides them into training, test and valudation sets in a 50/30/20 split. `file_names` is a list of paths to the videos, `out_path` is the output directory, `verbose` turns verbosity on/off.

`Cleanup(path = './data/', keep_annotated = True)` deletes training/validation/test sets. `Path` is the path to the sets (should be the same as the `out_path` argument from the `Extractor` function). `keep_annotated` specifies whether or not to keep `annotated.csv`.

From the command line, calling `src/frame_capture.py input_path out_path` will extract frames and build the datasets, where `input_path` is the path to the raw videos and `out_path` denotes where to store the datasets.

These functions assume the raw videos are organized in the same structure as in the original google drive.

##### Data.py
`data.py` defines the `Import_Data` class, which handles importing frames as tensors.

`Import_Data.__init__(train_set = 'original')`: Set to 'original' for original videos and 'masked' for the masked versions.

`Import_Data.get_data()` returns `X_train, y_train, X_test, y_test, X_val, y_val` from the specified data set.

`Import_data.get_labels(desired_set = 'train')` returns the ground truth labels for the specified set. One of 'train', 'test' or 'val'.

`Import_data.write_output(path,desired_set = 'train', y_pred)` is a convience function to write the ground truth labels and predicted labels to a csv. `path` speciefies the output csv path, `desired_set` is one of 'train', 'test', or 'val', and `y_pred` are the predicted labels from the output.

**Model Training**
Model training for original and masked videos is walked through in `Original_Model.ipynb` and `Masked_Model.ipynb` respectively. You will need Jupyter Notebook to take a look.

Outputs can be found in `output_original.csv` and `output_masked.csv`.

To train on your own, import `ThreeLayerConvNet`, `optimizer_init_fn` and `Import_Data` in a separate python or jupyter file. You can choose to import the functions in `frame_capture.py` or call the file from the command line to build datasets.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


 [dill]: <https://github.com/joemccann/dillinger>
 [git-repo-url]: <https://github.com/joemccann/dillinger.git>
 [john gruber]: <http://daringfireball.net>
 [df1]: <http://daringfireball.net/projects/markdown/>
 [markdown-it]: <https://github.com/markdown-it/markdown-it>
 [Ace Editor]: <http://ace.ajax.org>
 [node.js]: <http://nodejs.org>
 [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
 [jQuery]: <http://jquery.com>
 [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
 [express]: <http://expressjs.com>
 [AngularJS]: <http://angularjs.org>
 [Gulp]: <http://gulpjs.com>

 [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
 [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
 [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
 [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
 [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
 [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
