import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from contrastive_learning.models.resnet import ResNetSimCLR
from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.losses import get_contrastive_loss,get_supervised_loss
from contrastive_learning.train_contrastive import train_contrastive
from contrastive_learning.models.resnet import ResNetSimCLR
from neptunecontrib.monitoring.keras import NeptuneMonitor

def finetune(train_dataset, val_dataset, test_dataset, nb_classes,
                      contrastive_model, epochs=100, verbose=True, froze_backbone=True,neptune=None):
    
    # re-configure the model (remove the projection head)
    encoder = tf.keras.Model(contrastive_model.input, contrastive_model.outputs[0])
    # add a classification head
    predictions = Dense(nb_classes, activation="softmax")(encoder)
    model = tf.keras.Model(encoder.input, predictions)

    if verbose:
        model.summary()

    # fine-tune the model
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_accuracy',
                                                                   mode='max',
                                                                   save_best_only=True)
    neptune_callback = NeptuneMonitor()

    model.fit(train_dataset, validation=val_dataset, epochs=epochs, 
              callbacks=[model_checkpoint_callback, netptune_callback], verbose=verbose)
    model.load_weights(checkpoint_filepath)

    # get performance on test set
    test_accuracy = model.evaluate(test_dataset)[1]
    if verbose:
        print(f"Finetune test accuracy {test_accuracy}")
    
    neptune.log('finetune_test_accuracy', test_accuracy)

    return test_accuracy


if __name__ == "__main__":
    loss = get_supervised_loss()#get_contrastive_loss(temperature=1.0)
    batch_size = 32
    train, val, test = get_dataset('cifar_sup', batch_size=batch_size)
    input_shape = (224,224,3)
    out_dim = batch_size*2
    projector = ResNetSimCLR(input_shape,out_dim )
    projector.load_weights('ckpt/contrastive.h5')
    finetune(train,val,test,projector, tf.keras.optimizers.Adam(), epochs=100, verbose=True, froze_backbone=True)