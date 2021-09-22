import tensorflow as tf


def model_fit(x, y, loss_func, model, optimizer):
    with tf.GradientTape() as g:
        pred = model(x, training=True)
        loss = loss_func(pred, y)

    trainable_variables = model.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss