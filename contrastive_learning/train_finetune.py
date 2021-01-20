import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from contrastive_learning.models.resnet import ResNetSimCLR
from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.losses import get_contrastive_loss,get_supervised_loss
from contrastive_learning.train_contrastive import train_contrastive
from contrastive_learning.models.resnet import ResNetSimCLR

def finetune(train_dataset, val_dataset, test_dataset,
                      projector, loss_fn, optimizer,
                      epochs=100, verbose=True, froze_backbone=True):
    if verbose:
        projector.summary()
    
    # define the training and testing step
    @tf.function
    def train_step(batch_x):
        with tf.GradientTape() as tape:
            _, projection = projector(batch_x, training=True)
            loss = loss_fn(projection)
        grads = tape.gradient(loss, projector.trainable_weights)
        optimizer.apply_gradients(zip(grads, projector.trainable_weights))

        return loss

    @tf.function
    def test_step(batch_x):
        _, projection = projector(batch_x, training=False)
        loss = loss_fn(projection)

        return loss
    
    # training loop
    for epoch_i in range(epochs):
        # train iterations
        for batch_x in train_dataset:
            batch_loss = train_step(batch_x)
            if verbose:
                tf.print(f"[Epoch {epoch_i}] [Train] {batch_loss}")
        # test iterations
        for batch_x in test_dataset:
            batch_loss = test_step(batch_x)
            if verbose:
                tf.print(f"[Epoch {epoch_i}] [Test] {batch_loss}")
    
    # validation 
    for batch_x in val_dataset:
        batch_loss = test_step(batch_x)
        tf.print(f"[Epoch {epoch_i}] [Val] {batch_loss}")





if __name__ == "__main__":
    loss = get_supervised_loss()#get_contrastive_loss(temperature=1.0)
    batch_size = 32
    train, val, test = get_dataset('cifar_sup', batch_size=batch_size)
    input_shape = (224,224,3)
    out_dim = batch_size*2
    projector = ResNetSimCLR(input_shape,out_dim )
    projector.load_weights('ckpt/contrastive.h5')
    finetune(train,val,test,projector, tf.keras.optimizers.Adam(), epochs=100, verbose=True, froze_backbone=True)