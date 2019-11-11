import tensorflow as tf

def optimizer_init_fn(learning_rate):
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, nesterov = True)
    return optimizer