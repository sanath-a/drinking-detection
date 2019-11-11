import tensorflow as tf


class ThreeLayerConvNet(tf.keras.Model):
    '''
    A basic three layer Convolutional Neural Network built off the Keras API.  
    
    '''
    def __init__(self, num_classes = 2):
        super(ThreeLayerConvNet, self).__init__()
        '''
        Params
        -------
        num-classes: Number of target classes in dataset. Default is 2. 
        -------
        '''
        channel_1, channel_2 = 64, 64
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        self.conv1 = tf.keras.layers.Conv2D(channel_1,5, strides = 1,
                                            padding = "same",
                                            activation = "relu", kernel_initializer = initializer)
        self.conv2 = tf.keras.layers.Conv2D(channel_2, 3, strides = 1,
                                            padding = "same",
                                            activation = "relu", kernel_initializer = initializer)
        self.flatten = tf.keras.layers.Flatten()

        self.fc_1 = tf.keras.layers.Dense(num_classes, activation = "softmax", kernel_initializer = initializer)


    def call(self, x, training=False):
        scores = None

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        scores = self.fc_1(x)

        return scores
