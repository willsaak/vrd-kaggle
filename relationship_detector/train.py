import cv2
import keras
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple

import relationship_detector.config as config
from data.data_sequence import DataSequence
from models.inception_resnet_v2 import InceptionResNetV2

# backend.set_session(session=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

labels_csv = '../annotations/relationship_labels_and_none.csv'
imgs_path = '/mnt/renumics-research/datasets/vis-rel-data/img'

DOUBLE_IMAGE = True
SEPARATE_LABELS = False

VAL_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5


def batch_annotation_to_input(batch_annotation,
                              augmenter: Optional[keras.preprocessing.image.ImageDataGenerator] = None,
                              double_image: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    batch = [annotation_to_input(annotation, augmenter, double_image) for annotation in batch_annotation]
    batch_images = np.stack([annotation[0][0] for annotation in batch])
    batch_labels = np.stack([annotation[0][1] for annotation in batch])
    batch_classes = np.stack([annotation[1] for annotation in batch])
    return [batch_images, batch_labels], batch_classes


def annotation_to_input(annotation,
                        augmenter: Optional[keras.preprocessing.image.ImageDataGenerator] = None,
                        double_image: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    image = cv2.imread(annotation['image_name'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    ymin1 = int(annotation['ymin1'] * height)
    ymax1 = int(annotation['ymax1'] * height)
    xmin1 = int(annotation['xmin1'] * width)
    xmax1 = int(annotation['xmax1'] * width)
    ymin2 = int(annotation['ymin2'] * height)
    ymax2 = int(annotation['ymax2'] * height)
    xmin2 = int(annotation['xmin2'] * width)
    xmax2 = int(annotation['xmax2'] * width)
    ymin, ymax, xmin, xmax = min(ymin1, ymin2), max(ymax1, ymax2), min(xmin1, xmin2), max(xmax1, xmax2)
    if not double_image:
        cropped_image = np.zeros((ymax - ymin, xmax - xmin, channels), image.dtype)
        cropped_image[ymin1 - ymin:ymax1 - ymin, xmin1 - xmin:xmax1 - xmin] = image[ymin1:ymax1, xmin1:xmax1]
        cropped_image[ymin2 - ymin:ymax2 - ymin, xmin2 - xmin:xmax2 - xmin] = image[ymin2:ymax2, xmin2:xmax2]
    else:
        cropped_image = np.zeros((ymax - ymin, xmax - xmin, channels * 2), image.dtype)
        cropped_image[ymin1 - ymin:ymax1 - ymin, xmin1 - xmin:xmax1 - xmin, :channels] = image[ymin1:ymax1, xmin1:xmax1]
        cropped_image[ymin2 - ymin:ymax2 - ymin, xmin2 - xmin:xmax2 - xmin, channels:] = image[ymin2:ymax2, xmin2:xmax2]
    cropped_image = cv2.resize(cropped_image, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
    if augmenter is not None:
        cropped_image = augmenter.random_transform(cropped_image)
    normalized_image = cropped_image.astype(np.float32) / 127.5 - 1.

    if 'label' in annotation:
        label = keras.utils.to_categorical(annotation['label'], config.NUM_OF_LABELS)
    else:
        label1 = keras.utils.to_categorical(annotation['label1'], config.NUM_OF_FIRST_LABELS)
        label2 = keras.utils.to_categorical(annotation['label2'], config.NUM_OF_SECOND_LABELS)
        label = np.concatenate([label1, label2])
    klass = keras.utils.to_categorical(annotation['klass'], config.NUM_OF_CLASSES)
    return (normalized_image, label), klass


annotations = []
with open(labels_csv) as file:
    for line in file:
        line = line[:-1].split(',')
        image_name, klass, xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2 = (
            os.path.join(imgs_path, line[0] + '.jpg'), config.class_names.index(line[-1]),
            float(line[3]), float(line[4]), float(line[5]), float(line[6]),
            float(line[7]), float(line[8]), float(line[9]), float(line[10]))
        annotation = dict(image_name=image_name, klass=klass, xmin1=xmin1, xmax1=xmax1, ymin1=ymin1, ymax1=ymax1,
                          xmin2=xmin2, xmax2=xmax2, ymin2=ymin2, ymax2=ymax2)
        if SEPARATE_LABELS:
            annotation['label1'] = config.first_label_names.index(line[1])
            annotation['label2'] = config.second_label_names.index(line[2])
        else:
            annotation['label'] = config.label_names.index(','.join(line[1:3]))
        annotations.append(annotation)
train_annotations, val_annotations = train_test_split(annotations, test_size=VAL_SIZE)

train_augmenter = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
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
                               map_fn=lambda x: batch_annotation_to_input(x, train_augmenter, DOUBLE_IMAGE))
val_generator = DataSequence(val_annotations, batch_size=BATCH_SIZE, shuffle=False,
                             map_fn=lambda x: batch_annotation_to_input(x, None, DOUBLE_IMAGE))

model = InceptionResNetV2(
    include_top=True,
    weights='imagenet',
    labels=config.NUM_OF_LABELS if not SEPARATE_LABELS else config.NUM_OF_FIRST_LABELS + config.NUM_OF_SECOND_LABELS,
    classes=config.NUM_OF_CLASSES,
    input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3 if not DOUBLE_IMAGE else 6)
)

model.summary()

loss_checkpointer = keras.callbacks.ModelCheckpoint(
    filepath="../weights/relationship_classifier_best_model.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=EARLY_STOPPING_PATIENCE,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./graph',
    histogram_freq=0,
    write_graph=False,
    write_images=False
)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=1,
    callbacks=[loss_checkpointer, early_stopping, tensorboard_callback])

# Save the model
model.save('../weights/relationship_classifier_last.h5')
