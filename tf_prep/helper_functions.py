## dependencies
import os
import datetime
import zipfile 
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import TensorBoard as tb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.python.ops.check_ops import NUMERIC_TYPES

import matplotlib.pyplot as plt
##

## unzip data
def unzip_data(zip_file):  
  zip_ref = zipfile.ZipFile(zip_file) 
  zip_ref.extractall()
  zip_ref.close()
##

## walk through directory
def dir_walk(dir):
  for dirpath, dirnames, filenames in os.walk(dir):
    print(f"{len(dirnames)} directories and {len(filenames)} in '{dirpath}'.")
##

## Create TB callback
def create_tb_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tb_callback = tb(log_dir=log_dir)
  print(f"Saving TB log to '{log_dir}'")
  return tb_callback
##

## create a model from URL
def create_model(model_url, num_classes=2):
  """
  Create Sequential from Hub.
  Arguments:
    model_url (str): A TF Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer (number of target classes).
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extraction layer
    and Dense output layer with num_classes output neurons
  """

  # download pretrained model and save it as a keras layer
  feature_extraction_layer = hub.KerasLayer(model_url,
                                            trainable=False, # freeze params
                                            name="feature_extraction_layer",
                                            input_shape=IMAGE_SHAPE+(3,))
  #

  # Create Sequential
  model = tf.keras.Sequential([feature_extraction_layer,
                              layers.Dense(num_classes,
                                          activation = "softmax",
                                          name="output_layer")])

  return model
  #
##

## plot loss curves
def plot_loss(history,n_fig):
  """
  Plot training and validation loss and accuracy.
  Arguments:
    history: TensorFlow history object.
  
  Returns:
    plots of training/validation loss and accuracy  
  """

  # data
  history_ = history.history

  epochs = range(len(history_["loss"]))

  train_loss = history_["loss"]
  val_loss = history_["val_loss"]

  train_accuracy = history_["accuracy"]
  val_accuracy = history_["val_accuracy"]
  #

  fig, ax = plt.subplots(1,2,num=n_fig)
  # plot (matplotlib)
  ax[0].plot(epochs, train_loss,'b' , label="train_loss")
  ax[0].plot(epochs, val_loss, 'r' , label="val_loss")
  # ax[0].title('Loss')
  # ax[0].xlabel('Epochs')
  ax[1].plot(epochs, train_accuracy,'--b' , label="train_acc")
  ax[1].plot(epochs, val_accuracy, '--r' , label="val_acc")
  # ax[1].title('Accuracy')
  # ax[1].xlabel('Epochs')
  #

##
