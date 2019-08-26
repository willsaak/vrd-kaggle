import keras

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

IMAGE_SIZE = (299, 299)
num_of_classes = 23

image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
class_shape = (num_of_classes,)

image_input = keras.layers.Input(shape=image_shape)
class_input = keras.layers.Input(shape=class_shape)

net = InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=image_input,
    # input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    pooling='avg'
)

x = net(image_input)
x = keras.layers.Concatenate(x, class_input)
x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
