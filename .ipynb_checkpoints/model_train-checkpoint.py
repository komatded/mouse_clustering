from sklearn.model_selection import train_test_split
from data_generator import TripletGenerator
from model import create_model
import tensorflow as tf
import pandas as pd
import random
print(tf.config.list_physical_devices())

# Training configs
MIN_MOUSE_TRACK_LEN = 100
N_USERS_TO_TRAIN = 100
EMBEDDING_SIZE = 128
PAD_SIZE = 200
POSITIVES_PER_ANCHOR = 10
NEGATIVES_PER_ANCHOR = 10
TRAIN_EPOCHS = 5
DROP_TIME_LINE = False

# Load data
df = pd.read_pickle('./sw_139_data.pickle')
df = df[df.mouse_track.apply(len) >= MIN_MOUSE_TRACK_LEN]

# Filter users to train on
cookies = df.cookie.value_counts()
random.seed(420)
cookies = random.sample(list(cookies[cookies >= POSITIVES_PER_ANCHOR].keys()), k=N_USERS_TO_TRAIN)

# Dataset split
df = df[df.cookie.isin(cookies)]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=420)

# Data generators
TG = TripletGenerator(pad_size=PAD_SIZE,
                      positives_per_anchor=POSITIVES_PER_ANCHOR,
                      negatives_per_anchor=NEGATIVES_PER_ANCHOR,
                      drop_time_line=DROP_TIME_LINE)
train_triplet_generator, train_n_batches = TG.create_data_generator(train_df, batch_size=32)
test_triplet_generator, test_n_batches = TG.create_data_generator(test_df, batch_size=32)

# Model training
model = create_model(input_shape=(PAD_SIZE, 3 - DROP_TIME_LINE), embedding_size=EMBEDDING_SIZE)
model.layers[3].summary()
my_callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(x=train_triplet_generator, steps_per_epoch=train_n_batches,
          validation_data=test_triplet_generator, validation_steps=test_n_batches,
          epochs=TRAIN_EPOCHS, callbacks=my_callbacks)
