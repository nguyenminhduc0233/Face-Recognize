import keras
import tensorflow as tf

# Lớp trích xuất patch từ ảnh
@keras.saving.register_keras_serializable()
class PatchExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtractionLayer, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        channels = tf.shape(images)[3]
        
        patch_size = self.patch_size

        patches = tf.image.extract_patches(
            images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patch_shape = tf.shape(patches)
        num_patches = patch_shape[1] * patch_shape[2]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_size * patch_size * channels])
        
        return patches
