def train_loop(model_init,learning_rate,num_epochs, print_every, use = None, is_training = False):
    with tf.device(device):
        model = model_init(use)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        hist = {}
        iterations = []
        training_losses = []
        training_acc = []
        val_acc = []

        t = 0
        for epoch in range(num_epochs):
            train_loss.reset_states()
            train_accuracy.reset_states

            for x_np, y_np in train_dset:

                with tf.GradientTape() as tape:
                    scores = model(x_np, is_training)
                    loss = loss_fn(y_np, scores)

                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_np, scores)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            prediction = model(test_x, training=False)
                            t_loss = loss_fn(test_y, prediction)


                            val_loss.update_state(t_loss)
                            val_accuracy.update_state(test_y, prediction)




                        template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
                        print (template.format(t, epoch+1,
                                             train_loss.result(),
                                             train_accuracy.result()*100,
                                             val_loss.result(),
                                             val_accuracy.result()*100))
                        iterations.append(t)
                        training_losses.append(train_loss.result())
                        training_acc.append(train_accuracy.result() * 100)
                        val_acc.append(val_accuracy.result() * 100)
                    t += 1
        hist['iteration'] = iterations
        hist['loss'] = training_losses
        hist['train acc'] = training_acc
        hist['val acc'] = val_acc

        return model, hist
