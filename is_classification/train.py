import cv2
import keras
import numpy as np
import os
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from typing import Optional

from data.data_sequence import DataSequence
from is_classification.inception_resnet_v2 import InceptionResNetV2

# backend.set_session(
#     session=tf.Session(
#         config=tf.ConfigProto(
#             gpu_options=tf.GPUOptions(
#                 allow_growth=True))))

labels_csv = '../annotations/is_labels_and_none.csv'
imgs_path = '/mnt/renumics-research/datasets/vis-rel-data/img'

labels_names = ['/m/0cvnqh', '/m/01mzpv', '/m/080hkjn', '/m/0342h', '/m/02jvh9', '/m/04bcr3', '/m/0dt3t', '/m/01940j',
                '/m/071p9', '/m/03ssj5', '/m/04dr76w', '/m/01y9k5', '/m/026t6', '/m/05r5c', '/m/03m3pdh', '/m/01_5g',
                '/m/078n6m', '/m/01s55n', '/m/04ctx', '/m/0584n8', '/m/0cmx8', '/m/02p5f1q', '/m/07y_7']
classes_names = ['none', '/m/02gy9n', '/m/05z87', '/m/0dnr7', '/m/04lbp', '/m/083vt']

IMAGE_SIZE = (299, 299)
NUM_OF_LABELS = len(labels_names)
NUM_OF_CLASSES = len(classes_names)

VAL_SIZE = 0.2
BATCH_SIZE = 16


def batch_annotation_to_input(batch_annotation,
                              augmenter: Optional[keras.preprocessing.image.ImageDataGenerator] = None):
    batch = [annotation_to_input(annotation, augmenter) for annotation in batch_annotation]
    batch_images = np.stack([annotation[0][0] for annotation in batch])
    batch_labels = np.stack([annotation[0][1] for annotation in batch])
    batch_classes = np.stack([annotation[1] for annotation in batch])
    return [batch_images, batch_labels], batch_classes


def annotation_to_input(annotation, augmenter: Optional[keras.preprocessing.image.ImageDataGenerator] = None):
    image = cv2.imread(annotation['image_name'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('original', image)
    height, width = image.shape[:2]
    ymin = int(annotation['ymin'] * height)
    ymax = int(annotation['ymax'] * height)
    xmin = int(annotation['xmin'] * width)
    xmax = int(annotation['xmax'] * width)
    cropped_image = image[ymin:ymax, xmin:xmax]
    # cv2.imshow('cropped', cropped_image)
    cropped_image = cv2.resize(cropped_image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # cv2.imshow('resized', cropped_image)
    # TODO: augment image
    if augmenter is not None:
        cropped_image = augmenter.random_transform(cropped_image)
    # cv2.imshow('augmented', cropped_image)
    # cv2.waitKey(0)
    normalized_image = cropped_image.astype(np.float32) / 127.5 - 1.
    label = keras.utils.to_categorical(annotation['label'], NUM_OF_LABELS)
    klass = keras.utils.to_categorical(annotation['klass'], NUM_OF_CLASSES)
    return (normalized_image, label), klass


annotations = []
with open(labels_csv) as file:
    for line in file:
        line = line.split(',')
        image_name, label, klass, xmin, xmax, ymin, ymax = (
            os.path.join(imgs_path, line[0] + '.jpg'), labels_names.index(line[1]), classes_names.index(line[2]),
            float(line[3]), float(line[4]), float(line[5]), float(line[6]))
        annotations.append(dict(
            image_name=image_name, label=label, klass=klass, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        ))
train_annotations, val_annotations = train_test_split(annotations, test_size=VAL_SIZE)

train_augmenter = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=18,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=None,
    shear_range=0.05,
    zoom_range=0.05,
    fill_mode='constant',
    cval=0.0,
    horizontal_flip=True,
    rescale=None
)

train_generator = DataSequence(train_annotations, batch_size=BATCH_SIZE, shuffle=True,
                               map_fn=lambda x: batch_annotation_to_input(x, train_augmenter))
val_generator = DataSequence(val_annotations, batch_size=BATCH_SIZE, shuffle=False, map_fn=batch_annotation_to_input)

image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
model = InceptionResNetV2(include_top=True, weights='imagenet', labels=NUM_OF_LABELS, classes=NUM_OF_CLASSES,
                          input_shape=image_shape)

model.summary()

loss_checkpointer = ModelCheckpoint(
    filepath="../weights/is_classifier_best_model.hdf5",
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
tensorboard_callback = TensorBoard(
    log_dir='./graph',
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=1,
    callbacks=[loss_checkpointer, early_stopping, tensorboard_callback])

# Save the model
model.save('../weights/is_classifier_last.h5')
