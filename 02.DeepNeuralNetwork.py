import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", input_shape=[3]),
        layers.Dense(2, activation="relu"),
        layers.Dense(1),
    ]
)