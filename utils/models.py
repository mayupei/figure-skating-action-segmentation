import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Input,
    InputLayer,
    TimeDistributed,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from utils.config import get_config

CONFIG = get_config()


def build_lstm_classifier(seq_len, feature_dim, hidden_size, num_classes):
    model = Sequential(
        [
            Input(shape=(seq_len, feature_dim)),
            LSTM(hidden_size, return_sequences=False),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["model_stage1"]["learning_rate"]),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def masked_categorical_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(tf.reduce_max(y_true, axis=-1), 0), tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(tf.reduce_max(y_true, axis=-1), 0), tf.float32)
    matches = tf.cast(
        tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1)), tf.float32
    )
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def cnn_stage2(num_classes, embedding_dim, sequence_length):
    model = Sequential(
        [
            InputLayer(input_shape=(sequence_length,)),
            Embedding(
                input_dim=num_classes + 1,  # +1 for PAD_VALUE, mapped to last index
                output_dim=embedding_dim,
                mask_zero=False,
            ),  # Masking not supported with Conv1D
            Conv1D(
                filters=CONFIG["model_stage2"]["filters"],
                kernel_size=CONFIG["model_stage2"]["kernel_size"],
                padding="same",
                activation="relu",
            ),
            Dropout(CONFIG["model_stage2"]["dropout"]),
            Conv1D(
                filters=CONFIG["model_stage2"]["filters"],
                kernel_size=CONFIG["model_stage2"]["kernel_size"],
                padding="same",
                activation="relu",
            ),
            Dropout(CONFIG["model_stage2"]["dropout"]),
            TimeDistributed(Dense(num_classes, activation="softmax")),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["model_stage2"]["learning_rate"]),
        loss=masked_categorical_crossentropy,
        metrics=[masked_accuracy],
    )

    return model
