import keras

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2


def get_is_classifier(image_shape, num_of_labels, num_of_classes):
    image_input = keras.layers.Input(shape=image_shape)
    label_input = keras.layers.Input(shape=(num_of_labels,))

    net = InceptionResNetV2(
        include_top=False,
        weights=None,
        # input_tensor=image_input,
        input_shape=image_shape,
        pooling='avg'
    )

    x = net(image_input)
    x = keras.layers.Concatenate()([x, label_input])
    x = keras.layers.Dense(num_of_classes, activation='softmax', name='predictions')(x)

    model = keras.models.Model([image_input, label_input], x, name='is_classifier')

    return model

# IMAGE_SIZE = (299, 299)
# NUM_OF_LABELS = 23
# NUM_OF_CLASSES = 5
#
# image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
# label_shape = (NUM_OF_LABELS,)
#
# image_input = keras.layers.Input(shape=image_shape)
# label_input = keras.layers.Input(shape=label_shape)
#
# net = InceptionResNetV2(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=image_input,
#     # input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     pooling='avg'
# )
#
# x = net(image_input)
# x = keras.layers.Concatenate()([x, label_input])
# x = keras.layers.Dense(NUM_OF_CLASSES, activation='softmax', name='predictions')(x)
#
# model = keras.models.Model([image_input, label_input], x, name='is_classifier')
