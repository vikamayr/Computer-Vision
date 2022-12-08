import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

batch_size = 32
img_height = 180
img_width = 180
image_size = (img_height, img_width)
data_dir = pathlib.Path('PetImages')

epochs = 20

test_image = "PetImages/Cat/3.jpg"
my_dog = "leevi_dog.jpg"


def create_dataset(data_dir, dataset):
    created_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset=dataset,
                                                                       seed=123, image_size=(img_height, img_width),
                                                                       batch_size=batch_size)
    return created_data


def create_model():
    num_classes = 2
    model = Sequential([
        data_augmentation(),
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
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
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    return data_augmentation


def train_model(model, train_data, val_data):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]

    trained_model = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    return trained_model


def test_model(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()
    print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score[1]), 100 * score[1]))
    plt.imshow(img)
    plt.xlabel("cat: " + ("%.4f" % score[0]) + "\n" +
               "dog: " + ("%.4f" % score[1]))
    plt.show()


def visualize_dataset(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def visualize_training(trained_model):
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']

    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']

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


dataset_directory = pathlib.Path('PetImages')
train_ds = create_dataset(dataset_directory, "training")
val_ds = create_dataset(dataset_directory, "validation")
class_names = train_ds.class_names
print(class_names)

visualize_dataset(train_ds)

# print batch size
"""
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
"""

# Optional, depending of performance of computer
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = create_model()
trained_model = train_model(model, train_ds, val_ds)


visualize_training(trained_model)
test_model(model, test_image)
test_model(model, my_dog)
