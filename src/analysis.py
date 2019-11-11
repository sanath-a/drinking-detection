import tensorflow as tf
import numpy as np
from model import ThreeLayerConvNet
from trainer import train_loop
from data import import_data, Dataset
def custom_model_init():
    return ThreeLayerConvNet(128,64,2)

def optimizer_init():
     optimizer = tf.keras.optimizers.SGD(learning_rate, momentum = 0.9, nesterov = True)

     return optimizer
