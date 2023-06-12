from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from keras.applications import ResNet50, EfficientNetB0
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras import Model
import tensorflow as tf
import numpy as np
import logging
import os


def data_gen(dir, img_size, batch_size, classes, train):
    if train:
        img_gen = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=20,
                                     width_shift_range=0.25,
                                     height_shift_range=0.25,
                                     shear_range=0.25,
                                     zoom_range=0.25,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
    else:
        img_gen = ImageDataGenerator(rescale=1. / 255)

    return img_gen.flow_from_directory(batch_size=batch_size,
                                       directory=dir,
                                       shuffle=True,
                                       target_size=img_size,
                                       classes=classes,
                                       class_mode='categorical')


def init_cnn_model(class_num, input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    return model


def init_resnet_model(class_num, input_shape, fine_tuning):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    if fine_tuning:
        for layer in base_model.layers[:-10]:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(class_num, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def init_efficientnet_model(class_num, input_shape, fine_tuning):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    if fine_tuning:
        for layer in base_model.layers[:-10]:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(class_num, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def tensorboard_cb(model):
    log_dir = "logs/" + model
    callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    print("tensorboard_cb: " + log_dir)
    return callback


if __name__ == '__main__':
    n = 4

    i1 = 6
    i2 = 36
    i3 = 66

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    base_dir = os.path.join('food101')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    all_classes = sorted(os.listdir(train_dir))
    classes = [all_classes[i] for i in [i1, i2, i3]]

    print(f"classes: {classes}\n")

    TRAIN_SIZE = 750
    VAL_SIZE = 225
    CLASS_NUM = 3
    BATCH_SIZE = 25
    IMG_SIZE = (200, 200)
    INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], CLASS_NUM)
    EPOCHS = 10

    train_data_gen = data_gen(train_dir, IMG_SIZE, BATCH_SIZE, classes, train=True)
    val_data_gen = data_gen(validation_dir, IMG_SIZE, BATCH_SIZE, classes, train=False)

    cnn_model_adam = init_cnn_model(CLASS_NUM, INPUT_SHAPE)
    cnn_model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
    cnn_history_adam = cnn_model_adam.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
        callbacks=[tensorboard_cb('cnn_adam')]
    )

    resnet_model_adam = init_resnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=False)
    resnet_model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                              metrics=['accuracy'])
    resnet_history_adam = resnet_model_adam.fit(train_data_gen,
                                                steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
                                                epochs=EPOCHS,
                                                validation_data=val_data_gen,
                                                validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
                                                callbacks=[tensorboard_cb('resnet_adam')])

    resnet_model_sgd = init_resnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=False)
    resnet_model_sgd.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy',
                             metrics=['accuracy'])

    resnet_history_sgd = resnet_model_sgd.fit(train_data_gen,
                                              steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
                                              epochs=EPOCHS,
                                              validation_data=val_data_gen,
                                              validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
                                              callbacks=[tensorboard_cb('resnet_sgd')])

    resnet_model_adam_ft = init_resnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=True)
    resnet_model_adam_ft.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    resnet_history_adam_ft = resnet_model_adam_ft.fit(train_data_gen,
                                                      steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
                                                      epochs=EPOCHS,
                                                      validation_data=val_data_gen,
                                                      validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
                                                      callbacks=[tensorboard_cb('resnet_adam_ft')])

    efficientnet_model_adam = init_efficientnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=False)
    efficientnet_model_adam.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    efficientnet_history_adam = efficientnet_model_adam.fit(train_data_gen,
                                                            steps_per_epoch=int(
                                                                np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
                                                            epochs=EPOCHS,
                                                            validation_data=val_data_gen,
                                                            validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
                                                            callbacks=[tensorboard_cb('efficientnet_adam')])

    efficientnet_model_sgd = init_efficientnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=False)
    efficientnet_model_sgd.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy',
                                   metrics=['accuracy'])
    efficientnet_history_sgd = efficientnet_model_sgd.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
        callbacks=[tensorboard_cb('efficientnet_sgd')]
    )

    efficientnet_model_adam_ft = init_efficientnet_model(CLASS_NUM, INPUT_SHAPE, fine_tuning=True)
    efficientnet_model_adam_ft.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    efficientnet_history_adam_ft = efficientnet_model_adam_ft.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(TRAIN_SIZE / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(VAL_SIZE / float(BATCH_SIZE))),
        callbacks=[tensorboard_cb('efficientnet_adam_ft')]
    )
