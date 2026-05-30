import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


def build_model(num_classes):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    return model