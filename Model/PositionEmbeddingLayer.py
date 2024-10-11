import keras
import tensorflow as tf

# Lớp embedding cho position
@keras.saving.register_keras_serializable()
class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, num_patches, embedding_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.position_embeddings = self.add_weight(
            shape=(num_patches, embedding_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, positions):
        return tf.gather(self.position_embeddings, positions)