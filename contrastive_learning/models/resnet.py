

import tensorflow as tf


def ResNetSimCLR(input_shape, out_dim, pretrained=True, frozen_layer=True, base_model='resnet18'):
    inputs = tf.keras.layers.Input(shape=(input_shape))

    base_encoder = tf.keras.applications.ResNet50(include_top=False, weights='imagenet' if pretrained else None,
                                                  input_shape=input_shape, pooling='avg')
    base_encoder.training = not frozen_layer
    h = base_encoder(inputs)

    # projection head
    x = tf.keras.layers.Dense(units=out_dim)(h)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=out_dim)(x)

    return tf.keras.Model(inputs=inputs, outputs=[h, x])