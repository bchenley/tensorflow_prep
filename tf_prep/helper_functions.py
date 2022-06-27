## dependencies
import os
import datetime
from zipfile import ZipFile
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import callbacks 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental import preprocessing as experimental_preprocessing
from tensorflow.keras import Sequential
from tensorflow.python.ops.check_ops import NUMERIC_TYPES

import numpy as np
import matplotlib.pyplot as plt
##

## unzip data
def unzip_data(zip_file):  
  zip_ref = ZipFile(zip_file) 
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
  tb_callback = callbacks.TensorBoard(log_dir=log_dir)
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

## get image training and test data using tf.keras.preprocessing.image_dataset_from_directory
def image_train_test_from_directory(train_dir,test_dir,image_size,label_mode,batch_size):
  train_data = tf.keras.preprocessing.image_dataset_from_directory(directory = train_dir,
                                                                   image_size = image_size,
                                                                   label_mode = label_mode,
                                                                   batch_size = batch_size)

  test_data = tf.keras.preprocessing.image_dataset_from_directory(directory = test_dir,
                                                                  image_size = image_size,
                                                                  label_mode = label_mode,
                                                                  batch_size = batch_size)
  return train_data, test_data
##

## create model from base
def fit_base_model(train_data, test_data, 
                     base_model, base_model_trainable, 
                     input_shape, aug,
                     num_outputs, output_activation,
                     loss, optimizer, metrics, 
                     epochs, initial_epoch, 
                     pct_validate,
                     callback):

  for layer,trainable in zip(base_model.layers,base_model_trainable):
      layer.trainable = trainable

  # 1. create inputs into our model
  inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
  #

  # 2. augment features
  if aug:
    inputs = aug(inputs)
  #
  
  # 3. pass inputs to base model
  h_base = base_model(inputs)
  #

  print(f"shape of base model output: {h_base.shape}")

  # 4. Average pool outputs
  h_avg = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(h_base)
  #

  print(f"shape of average pooling output: {h_avg.shape}")

  # 5. create output activation layer
  outputs = tf.keras.layers.Dense(num_outputs, activation=output_activition, name = "output_layer")(h_avg)
  #

  # 6. create main model
  model = tf.keras.Model(inputs, outputs)
  #
  
  # 7. compile model
  model.compile(loss = loss,
                optimizer = optimizer,
                metrics = metrics)
  #

  # 8. fit model
  history = model.fit(train_data,
                      epochs = epochs,
                      initial_epoch = initial_epoch,
                      steps_per_epoch = len(train_data),
                      validation_data = test_data,
                      validation_steps = int(pct_validate * len(test_data)),
                      callbacks = callback)
  #
  return model, history
##

def finetune_model(train_data, test_data, 
                     model,                      
                     loss, optimizer, metrics, 
                     epochs, initial_epoch, 
                     pct_validate,
                     callback):

  # 1. compile model
  model.compile(loss = loss,
                optimizer = optimizer,
                metrics = metrics)
  #

  # 2. fit model
  history = model.fit(train_data,
                      epochs = epochs,
                      initial_epoch = initial_epoch,
                      steps_per_epoch = len(train_data),
                      validation_data = test_data,
                      validation_steps = int(pct_validate * len(test_data)),
                      callbacks = callback)
  #
  return model, history
##

def compare_histories(hist1, hist2, initial_epochs):
    '''
    Compare two TF histories
    '''

    acc1, loss1, val_acc1, val_loss1 = hist1.history["accuracy"], hist1.history["loss"], hist1.history["val_accuracy"], hist1.history["val_loss"]
    acc2, loss2, val_acc2, val_loss2 = hist2.history["accuracy"], hist2.history["loss"], hist2.history["val_accuracy"], hist2.history["val_loss"]

    acc12, val_acc12 = acc1 + acc2, val_acc1 + val_acc2
    loss12, val_loss12 = loss1 + loss2, val_loss1 + val_loss2

    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.plot(acc12, label="train acc")
    plt.plot(loss12, label="trian loss")

    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Begin Fine Tuning')
    plt.legend(loc='lower right')
    plt.title("Training and Validation")

    plt.subplot(2, 1, 2)
    plt.plot(val_acc12, label="val acc")
    plt.plot(val_loss12, label="val loss")

    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Begin Fine Tuning')
    plt.legend(loc='lower right')
    plt.title("Training and Validation")

    return acc12, val_acc12, loss12, val_loss12
