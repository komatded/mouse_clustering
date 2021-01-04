from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(embedding_anchor, embedding_positive, embedding_negative):
    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(embedding_anchor * embedding_positive, axis=-1, keepdims=True) -
        K.sum(embedding_anchor * embedding_negative, axis=-1, keepdims=True))
    return loss


def create_base_model(input_shape, embedding_size):
    input_layer = Input(shape=input_shape)
    x = BatchNormalization()(input_layer)
    x = LSTM(100, return_sequences=False)(x)  # dropout=0.1, recurrent_dropout=0.1
    x = Dense(embedding_size, activation='relu')(x)
    base_network = Model(inputs=input_layer, outputs=x)
    return base_network


def create_model(input_shape, embedding_size):
    base_model = create_base_model(input_shape, embedding_size)

    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    embedding_anchor = base_model([input_anchor])
    embedding_positive = base_model([input_positive])
    embedding_negative = base_model([input_negative])

    loss = bpr_triplet_loss(embedding_anchor, embedding_positive, embedding_negative)
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam())
    return model
