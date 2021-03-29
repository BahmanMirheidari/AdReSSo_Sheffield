from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import TimeDistributed, Input, Flatten, MaxPool1D, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from src.conf import *

import keras.backend as K
class AttLayer(layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        super(AttLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.init = initializers.get('normal')
    def get_config(self):
        config = super(AttLayer, self).get_config()
        config.update({
            'attention_dim': self.attention_dim,
            'init': self.init,
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = tf.Variable(self.init((input_shape[-1], self.attention_dim)), name = 'w')
        self.b = tf.Variable(self.init((self.attention_dim, )), name = 'b')
        self.u = tf.Variable(self.init((self.attention_dim, 1)), name = 'u')
        self.epsilon = tf.constant(value=0.000001, shape=input_shape[1], name = 'epsilon')
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask


    def call(self, inputs, mask=None):
        uit = tf.math.tanh(tf.nn.bias_add(tf.matmul(inputs, self.W), self.b))
        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, -1)
        ait = tf.math.exp(ait)
        ait /= tf.cast(tf.math.reduce_sum(ait, axis=1, keepdims=True) + self.epsilon, dtype=tf.float32)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.math.reduce_sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def Transformer_Att_model(embedding_layer,transformer_block):
    sequence_input = layers.Input(shape=(MAX_TRAN_LEN,), dtype='float32')
    x = embedding_layer(sequence_input)
    x = transformer_block(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = layers.Dropout(drop_rate)(x)
    x = AttLayer(Att_DIM,name = 'sent_attention')(x)
    #x = layers.Dense(model_item.Dense_DIM, activation='relu',name = 'dense')(x)
    preds = layers.Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(l2))(x)
    model = keras.Model(sequence_input, preds)
    print(model.summary())
    return model


def BiRNN_Att_model(embedding_layer):
    sequence_input = Input(shape=(MAX_TRAN_LEN,), dtype='int32')
    x = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(drop_rate)(x)
    x = AttLayer(Att_DIM, name = 'attention')(x)
    preds = Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(l2))(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    return model

def BiRNN_feat_model():
    feats_input = Input(shape=(MAX_TRAN_LEN, Feats_dim), dtype='float32')
    
    x = Bidirectional(LSTM(LSTM_DIM[0], return_sequences=True), merge_mode='sum')(feats_input)
    x = Bidirectional(LSTM(LSTM_DIM[0], return_sequences=True), merge_mode='sum')(x)
    
    #x = Conv1D(cnn_N_filt[2], cnn_len_filt[2], strides=1, padding='valid')(x)
    #x = MaxPool1D(pool_size=cnn_max_pool_len[2])(x)
    #x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    
    #x = Dropout(drop_rate)(x)
    
    x = AttLayer(Att_DIM, name = 'attention')(x)
    preds = Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(l2))(x)
    model = Model(feats_input, preds)
    print(model.summary())
    return model

def Hie_Att_model(embedding_layer):
    ############word-level#############
    sentence_input = Input(shape=(MAX_SENT_LEN,), dtype='int32')
    x = embedding_layer(sentence_input)
    x = Dropout(drop_rate)(x)
    x = Bidirectional(LSTM(LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(Dense_DIM, activation='relu')(x)
    output = AttLayer(Att_DIM,name = 'word_attention')(x)
    sentEncoder = Model(sentence_input, output)
    print(sentEncoder.summary())
    ############sentence-level: Bi-rnn############
    review_input = Input(shape=(MAX_SENT_NUM, MAX_SENT_LEN), dtype='int32')
    x = TimeDistributed(sentEncoder)(review_input)
    x = Bidirectional(GRU(LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(drop_rate)(x)
    x = AttLayer(Att_DIM,name = 'sent_attention')(x)
    preds = Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(l2))(x)
    model = Model(review_input, preds)
    print(model.summary())
    return model


def Hie_feat_model():
    ############word-level#############
    sentence_input = Input(shape=(MAX_SENT_LEN,Feats_dim), dtype='float32')
    x = Bidirectional(LSTM(LSTM_DIM, return_sequences=True), merge_mode='sum')(sentence_input)
    x = Dropout(drop_rate)(x)
    x = Dense(Dense_DIM, activation='relu')(x)
    output = AttLayer(Att_DIM,name = 'word_attention')(x)
    sentEncoder = Model(sentence_input, output)
    print(sentEncoder.summary())
    ############sentence-level: Bi-rnn############
    feats_input = Input(shape=(MAX_SENT_NUM, MAX_SENT_LEN, Feats_dim), dtype='float32')
    x = TimeDistributed(sentEncoder)(feats_input)
    x = Bidirectional(GRU(LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(drop_rate)(x)
    x = AttLayer(Att_DIM,name = 'sent_attention')(x)
    preds = Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(l2))(x)
    model = Model(feats_input, preds)
    print(model.summary())
    return model



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
    def build(self, input_shape):

        super(TransformerBlock, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, trainable=True)
        self.embed_dim = embed_dim
    def build(self, input_shape):
        super(TokenAndPositionEmbedding, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)