from sklearn.model_selection import train_test_split
from data_generator import TripletGenerator
from model import create_model
import tensorflow as tf
import pandas as pd
import random

MIN_MOUSE_TRACK_LEN = 50
N_USERS_TO_TRAIN = 500
EMBEDDING_SIZE = 128
PAD_SIZE = 200
POSITIVES_PER_ANCHOR = 15
NEGATIVES_PER_ANCHOR = 15

random.seed(420)

print(tf.config.list_physical_devices())

df = pd.read_pickle('./sw_139_data.pickle')
df = df[df.mouse_track.apply(len) >= MIN_MOUSE_TRACK_LEN]
cookies = df.cookie.sample(n=N_USERS_TO_TRAIN, random_state=420)
df = df[df.cookie.isin(cookies)]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=420)

TG = TripletGenerator(pad_size=PAD_SIZE,
                      positives_per_anchor=POSITIVES_PER_ANCHOR,
                      negatives_per_anchor=NEGATIVES_PER_ANCHOR)
train_triplet_generator, train_n_batches = TG.create_data_generator(train_df, batch_size=32)
test_triplet_generator, test_n_batches = TG.create_data_generator(test_df, batch_size=32)

model = create_model(input_shape=(PAD_SIZE, 3), embedding_size=EMBEDDING_SIZE)
model.summary()
model.layers[3].summary()

my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
                tf.keras.callbacks.TensorBoard(log_dir='./logs')]

model.fit(x=train_triplet_generator, steps_per_epoch=train_n_batches,
          validation_data=test_triplet_generator, validation_steps=test_n_batches,
          epochs=3, callbacks=my_callbacks)
