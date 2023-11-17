#%%
# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
# %%
# Import a split dataset
# Define your destination directories
train_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\train"
validation_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\validation"
test_dir = r"C:\Users\ACER\Documents\SHRDC_AI\8) Capstone\1) CV\Split_Dataset\test"

BATCH_SIZE = 64
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True,batch_size=BATCH_SIZE,
image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=BATCH_SIZE,
image_size=IMG_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,shuffle=True,batch_size=BATCH_SIZE,
image_size=IMG_SIZE)
# %%
# Inspect some data example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.grid('off')
# %%
# Convert the tenserflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
# Defina a layer for data normalization
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# %%
# Transfer learning model
# Data augmentation layer > preprocess input > transfer learning model

#(A) Load the pretrained model using keras.applications
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()
keras.utils.plot_model(base_model)
# %%
#(B) Freeze the entire feature extractor
base_model.trainable = False
base_model.summary()
# %%
#(C) Create global average poolong player
global_avg = keras.layers.GlobalAveragePooling2D()
#(D) Create the output layer
output_layer = keras.layers.Dense(len(class_names), activation='softmax')
#(E) Build the entire pipeline using Functional API
#a. Inputs
inputs = keras.Input(shape=IMG_SHAPE)
#b. Data normalization
x = preprocess_input(inputs)
#c. Transfer learning feature extractor
x = base_model(x,training=False)
#d. Calssification layers
x = global_avg(x)
x = keras.layers.Dropout(0.3)(x)
outputs = output_layer(x)
#g. Build the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
# %%
# Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# %%
# Create a tensorboard callback project
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)
# %%
# Evaluate the model before training
model.evaluate(test_dataset)
# %%
# Model training
early_stopping = callbacks.EarlyStopping(patience=2)
EPOCHS = 10
history = model.fit(train_dataset,       validation_data=validation_dataset, epochs=EPOCHS, callbacks=[tb,early_stopping])
# %%
# Plot loss performance graph
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
# %%
# Plot accuracy performance graph
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend()
plt.show()
# %%
# Evaluate the model
model.evaluate(test_dataset)
# %%
# Deployment
#(A) Retrive a batch of images from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
# %%
#(B) Display result in matplotlib
prediction_indexes = np.argmax(predictions, axis=1)
# %%
# Create a label map for the classes
label_map = {i:names for i,names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_indexes]
label_class_list = [label_map[i] for i in label_batch]

plt.figure(figsize=(15,15))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Label:{label_class_list[i]}, Prediction:{prediction_label[i]}")
    plt.axis('off')
    plt.grid('off')
# %%
# Model save
model.save(os.path.join('concrete_classify.h5'))
