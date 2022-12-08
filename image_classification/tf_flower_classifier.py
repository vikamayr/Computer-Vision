import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np

"""
This notebook follows tensorflow Image classification notebook, which can be found from:
https://www.tensorflow.org/tutorials/images/classification
"""

# Global variables
batch_size = 32
img_height = 180
img_width = 180
epochs = 20


def configure_performance(train_ds, val_ds):
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


def create_dataset(data_directory, dataset_name):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset=dataset_name,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return dataset


def create_model(augmented_data):
    num_classes = 5
    model = Sequential([
        augmented_data,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model


def data_augmentation():
    augmented_data = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    return augmented_data


def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    return data_dir


def make_predictions(model):
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    img = keras.preprocessing.image.load_img(sunflower_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))


def train_model(model, train_data, val_data):
    trained_model = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    return trained_model


def visualize_data(train_dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def visualize_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


"""
Setup the dataset
"""
dataset_directory = download_dataset()
train_ds = create_dataset(dataset_directory, "training")
val_ds = create_dataset(dataset_directory, "validation")
class_names = train_ds.class_names
visualize_data(train_ds)

"""
The image_batch is a tensor of the shape (32, 180, 180, 3). 
This is a batch of 32 images of shape 180x180x3 (the last dimension refers to color channels RGB).
The label_batch is a tensor of the shape (32,), these are corresponding labels to the 32 images.
"""
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

"""
Configure performance
"""
train_ds, val_ds = configure_performance(train_ds, val_ds)

"""
Create and train the model
"""
data_augmentation = data_augmentation()
model = create_model(data_augmentation)
trained_model = train_model(model, train_ds, val_ds)

"""
Visualize training results
"""
visualize_training(trained_model)

"""
Make predictions with image not included in dataset
"""
make_predictions(model)
