#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 01:02:16 2020

@author: rajiv
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import time

image_size = (256,256)
batch_size = 32
# Generate the dataset from the directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/rajiv/Documents/experiments/chest_xray/train', labels='inferred', label_mode='binary', color_mode='rgb',
    batch_size=batch_size, image_size=(256,256), validation_split=0.2, subset='training', seed = 1284)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/rajiv/Documents/experiments/chest_xray/train', labels='inferred', label_mode='binary', color_mode='rgb',
    batch_size=32, image_size=(256,256), validation_split=0.2, subset='validation', seed = 1284)

# plot to see how the dataset looks like
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(int(labels[i]))
        plt.axis('off')
        
# When we don't have a large image dataset, it's a good practice to artificially introduce sample diversity by
# Applying random yet realistic transformations to the training images, such as horizontal flip or small random
# rotations. This helps expose the model to a different aspects of the training data while slowing overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ])

# Visualizing the augmented dataset
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype('uint8'))
    plt.axis('off')


#Buffered prefetching so we can yield data from disk without having I/O becoming blocking:
train_ds = train_ds.prefetch(buffer_size = 32)
val_ds = val_ds.prefetch(buffer_size = 32)


    
# The below model is a Xception network 
def custom_model(input_shape, num_classes):
    # Data Augmentation is implemented as a part of the model, This implementation will benefit from GPU acceleration 
    inputs = tf.keras.Input(shape = input_shape)
    x = data_augmentation(inputs)
    
    # Entry Block
    x = layers.experimental.preprocessing.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    previous_block_activation = x
    
    for size in [128, 256, 512, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding ='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(3, strides = 2, padding= 'same')(x)
        
        # project residual 
        residual = layers.SeparableConv2D(
            size, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        
    x = layers.SeparableConv2D(1024,3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
  
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
      activation = 'sigmoid'
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
  
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = custom_model(input_shape=image_size + (3,), num_classes=2)
#keras.utils.plot_model(model, show_shapes=True)


epochs = 100  # In practice you will need at least 50 epochs

callbacks = [
  keras.callbacks.ModelCheckpoint('save_at_{epoch}.h5'),
]
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_start_time = time.time()
history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
train_stop_time = time.time()
total_time = train_stop_time - train_start_time
print("The total time taken to train the model is : ", total_time)

