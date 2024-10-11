import tensorflow as tf
from keras import layers, models, callbacks, Input, datasets, utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np
from PatchExtractionLayer import PatchExtractionLayer
from PatchEmbeddingLayer import PatchEmbeddingLayer
from PositionEmbeddingLayer import PositionEmbeddingLayer

# Hàm cộng patch embeddings và position embeddings
def add_patch_and_position_embeddings(patch_embeddings, position_embeddings):
    return patch_embeddings + position_embeddings


def transfomer_encoder_mlp_head(input_shape, num_labels, patch_size, num_patches, embedding_dim):

    inputs = layers.Input(shape=input_shape)

    # Tạo patch embeddings
    patch_extraction_layer = PatchExtractionLayer(patch_size)
    patch_extraction = patch_extraction_layer(inputs)
    
    patch_embedding_layer = PatchEmbeddingLayer(embedding_dim)
    patch_embeddings = patch_embedding_layer(patch_extraction)
    
    # Tạo position embeddings
    position_embedding_layer = PositionEmbeddingLayer(num_patches, embedding_dim)
    positions = tf.range(num_patches)  # Giả sử các vị trí từ 0 đến num_patches-1
    position_embeddings = position_embedding_layer(positions)
    
    # Cộng patch embeddings và position embeddings
    combined_embeddings = add_patch_and_position_embeddings(patch_embeddings, position_embeddings)


    # Áp dụng Layer Normalization
    layer_norm = layers.LayerNormalization(epsilon=1e-6)
    normalized_patches = layer_norm(combined_embeddings)

    num_kernel = normalized_patches.shape[-1]

    # Khởi tạo một lớp MultiHeadAttention
    mha = layers.MultiHeadAttention(num_heads=8, key_dim=num_kernel//8)

    # Trong trường hợp này, chúng ta sử dụng cùng một tensor làm query, key, và value
    x = mha(query = normalized_patches, key = normalized_patches, value = normalized_patches)

    mix = layers.Add()([x, combined_embeddings])

    x = layers.LayerNormalization()(mix)

    # Thêm MLP
    x = layers.Dense(units=num_kernel, activation='relu')(x)

    output_2 = x

    output_3 = layers.Add()([output_2, mix])

    # MLP Head

    output_3 = layers.GlobalAveragePooling1D()(output_3)
    
    output_3 = layers.Dropout(0.2)(output_3)

    # Dự đoán các lớp đích
    outputs = layers.Dense(units=num_labels, activation="softmax")(output_3)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def train_model(train_dataset, val_dataset, num_labels, input_shape, epochs, patch_size, num_patches, embedding_dim):
    model = transfomer_encoder_mlp_head(input_shape=input_shape, num_labels=num_labels, patch_size=patch_size, num_patches=num_patches, embedding_dim=embedding_dim)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

    return model, history

def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show() 

def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)

    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(true_labels, predicted_labels))

def main():
    batch_size = 32
    epochs = 300
    num_labels = 351
    input_shape = (160, 160, 3)
    image_size = (160, 160)
    patch_size = 8
    num_patches = 400
    embedding_dim = 128
    name_model = 'vit_model_final_v29.keras'
    train_directory = r"data\extra_data_train"
    validation_directory = r"data\extra_data_val"
    test_directory = r"data\extra_data_test"

    # Tạo dataset cho dữ liệu huấn luyện
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size)

    # Tạo dataset cho dữ liệu validation
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_directory,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size)

    # Tạo dataset cho dữ liệu test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size)

    model, history = train_model(train_dataset, val_dataset, num_labels, input_shape, epochs, patch_size, num_patches, embedding_dim)

    # Lưu mô hình sau khi huấn luyện
    model.save(name_model)

    plot_history(history)

    evaluate_model(model, test_dataset)

if __name__ == '__main__':
    main()