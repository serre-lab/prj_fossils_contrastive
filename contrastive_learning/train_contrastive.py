import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def train_contrastive(train_dataset, val_dataset, test_dataset,
                      projector, loss_fn, optimizer,
                      epochs=100, verbose=True, froze_backbone=True,neptune=None):

    if verbose:
        projector.summary()
    
    # define the training and testing step
    train_accuracy = tf.metrics.Accuracy()
    test_accuracy = tf.metrics.Accuracy()
    @tf.function
    def train_step(batch_x):
        with tf.GradientTape() as tape:
            h_enc, projection = projector(batch_x, training=True)
            loss, logits, labels = loss_fn(projection)
            train_accuracy.update_state(tf.argmax(logits, -1), tf.argmax(labels, -1))
        grads = tape.gradient(loss, projector.trainable_weights)
        optimizer.apply_gradients(zip(grads, projector.trainable_weights))

        return loss,h_enc
        
    @tf.function
    def test_step(batch_x):
        h_enc, projection = projector(batch_x, training=False)
        loss, logits, labels = loss_fn(projection)
        test_accuracy.update_state(tf.argmax(logits,-1),tf.argmax(labels,-1))

        return loss,h_enc
    
    # training loop
    for epoch_i in range(epochs):
        # train iterations
        for batch_x in train_dataset:
            batch_loss, encoder = train_step(batch_x)
            neptune.log_metric("train_loss", batch_loss.numpy())
            neptune.log_metric("train_accuracy", train_accuracy.result().numpy()
            if verbose:
                tf.print(f"[Epoch {epoch_i}] [Train] {batch_loss} [Accuracy] {train_accuracy.result()}")
        # test iterations
        for batch_x in test_dataset:
            batch_loss, encoder = test_step(batch_x)
            neptune.log_metric("train_loss", batch_loss.numpy())
            neptune.log_metric("test_accuracy", test_accuracy.result().numpy())
            if verbose:
                tf.print(f"[Epoch {epoch_i}] [Test] {batch_loss}")
    
    # validation 
    for batch_x in val_dataset:
        batch_loss,encoder = test_step(batch_x)
        tf.print(f"[Epoch {epoch_i}] [Val] {batch_loss}")
    

    return projector
