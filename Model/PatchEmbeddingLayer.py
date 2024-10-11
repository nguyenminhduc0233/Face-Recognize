import keras
import tensorflow as tf

# Lớp embedding cho patch
@keras.saving.register_keras_serializable()
class PatchEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(PatchEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def call(self, patches):
        return self.dense(patches)