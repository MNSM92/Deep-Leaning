from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.models import load_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# history = model.fit(train_images, train_labels, epochs=5, batch_size=512)
# import model
loaded_model = load_model('model.h5')
# print history
# print(model.history.history)
# save the model
# model.save("model.h5")

test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, batch_size=512)
print(f"Test accuracy: {test_acc:.3f}", test_loss)
