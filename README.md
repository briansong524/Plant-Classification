# Plant-Classification


## Summary

Train a CNN model to classify images to one of twelve plants. Used Kaggle's [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) data to train and run the model. Created a simple Flask model to show the model in action. 

## Dependencies

*Python 2.7* (works for Python 3.x as well)

*pandas* - DataFrame for output 

*numpy* - To have images as arrays

*glob* - Handles directories

*skimage* - Reads image data and can handle converting to a resized array

*keras* - High level API to build neural networks. Tensorflow backend.

*Flask* - Lightweight API to host a model

## Files

**kaggle plant classification.ipynb** - Jupyter Notebook to train a model and save it locally

**flask_script.py** - Hosts a Flask API which can be accessed in browser

**model/model_weight_SGD.hdf5** - A trained CNN model to skip training if necessary

**templates/generate_page.html** - A simple HTML script that serves as a webpage template for Flask 


## How to use

### Train

First, get the training data from [here](https://www.kaggle.com/c/plant-seedlings-classification/data) and extract/store it in a folder "train" in the same path as where the Jupyter Notebook is located (otherwise, will have to change the variable 'work_dir' in the second block accordingly.

Running everything except the last two blocks will provide a trained model (obtained 95.34% testing accuracy with current CNN parameters). 

The last two blocks will export a csv of the predictions of the test set from Kaggle, if needed.

### Running Flask

Open flask_script.py first and change lines 34-39 to the paths that make sense to run on the computer. 

In Linux terminal, run:

`python flask_script.py`

to host the Flask server. I made a simple HTML page as a template to easily interact with the model and server. 

The server is set to run on http://0.0.0.0:8986/generate?imageid= (where 0.0.0.0 should be replaced with your IP address), so can type that in the browser to access the page when the server is running.

The page will show a picture, and the prediction of the plant. It is currently set to read the test image, so I don't really know whether the predictions are correct or not (not as easy as saying a picture is an image of a cat/dog). But it is supposedly 95% accurate so the predictions are for the most part correct. There is a generate button to reload another random image (or type in the ImageID to go to a specific image).

I built this Flask script and HTML page so that it can be easily configured to host different models. 


