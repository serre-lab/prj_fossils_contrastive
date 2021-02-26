import tensorflow as tf

from contrastive_learning.data.extant import get_supervised, decode_example
from contrastive_learning.data.extant import NB_CLASSES

size = 128
batch_size = 256
epochs = 100
nb_classes = NB_CLASSES

# train_ds, test_ds = get_supervised(batch_size, size)



raw_image_dataset = tf.data.TFRecordDataset('/media/data_cifs/projects/prj_fossils/data/processed_data/tf_records_2021_v1/images_*.tfrecords')
train_ds = raw_image_dataset.map(decode_example, num_parallel_calls=-1)



backbone = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(size, size, 3)))

x = backbone.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(200, activation="relu")(x)
logits = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=4, min_lr=0.00001)
history = model.fit(train_ds,
                    use_multiprocessing=True,
                    callbacks=[reduce_lr]) 
