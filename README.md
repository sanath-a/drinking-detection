# Drinking Detection for HABits Lab

This project is meant for the HABits lab at Northwestern University.

# Overview
This project aims to use CNNs to detect when an actor wearing a camera is drinking. Actual videos are not included in this repository - only the scripts and models used to analyze them.

# Using this Repository
**Dependencies**
OpenCV2 is used for frame capturing.
Tensorflow is used for Neural Networks.
Numpy and Pandas are used for data wrangling.
These dependencies are listed in `requirements.txt`

**Building the Dataset**
To build new train/test/val datasets, call `src/frame_capture.py` from the command line. This function randomly splits the data into 50/30/20 training, validation and testing sets. The path to the videos can be changed by modifying line 150 in the script.

**Model Training**
Model training for original and masked videos is walked through in `Original_Model.ipynb` and `Masked_Model.ipynb` respectively. You will need Jupyter Notebook to take a look.


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
