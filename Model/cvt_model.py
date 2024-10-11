import tensorflow as tf
from keras import models, Input, layers, optimizers
import matplotlib.pyplot as plt
from keras import callbacks
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Tạo mô hình CvT
def create_cvt_model(input_shape, num_labels):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    xm = layers.MaxPooling2D(2, 2)(x)
    
    x = layers.Flatten()(xm) 
    x = layers.LayerNormalization()(xm)

    # Reshape output_vector
    output_tensor = layers.Reshape((xm.shape[1], xm.shape[2], xm.shape[3]))(x)

    num_kernel = 128
    x1 = layers.Conv2D(num_kernel, kernel_size=3, strides=1, padding="same", activation='relu')(output_tensor)

    # Reshape tensor từ (num_batch, new_height, new_width, num_kernel) thành (num_batch, sequence_length, num_kernel)
    new_height, new_width = x1.shape[1], x1.shape[2]
    sequence_length = new_height * new_width
    reshaped_x1 = layers.Reshape((num_kernel, sequence_length))(x1) # reshaped_x1 = layers.Reshape((sequence_length, num_kernel))(x1)

    # Khởi tạo một lớp MultiHeadAttention
    mha = layers.MultiHeadAttention(num_heads=8, key_dim=sequence_length//8) # mha = layers.MultiHeadAttention(num_heads=8, key_dim=num_kernel//8)

    # Trong trường hợp này, chúng ta sử dụng cùng một tensor làm query, key, và value
    x = mha(query = reshaped_x1, key = reshaped_x1, value = reshaped_x1)

    # Reshape lại từ (num_batch, num_kernel, sequence_length) về (num_batch, new_height, new_width, num_kernel)
    x = layers.Reshape((new_height, new_width, num_kernel))(x)

    mix = layers.Add()([x, x1]) # [x,output_tensor]

    x = layers.LayerNormalization()(mix)

    # Thêm MLP
    x = layers.Dense(units=num_kernel, activation='relu')(x)

    output_2 = x

    output_3 = layers.Add()([output_2, mix])

    # MLP Head

    output_3 = layers.GlobalAveragePooling2D()(output_3)
    
    output_3 = layers.Dropout(0.2)(output_3)

    # Dự đoán các lớp đích
    outputs = layers.Dense(units=num_labels, activation="softmax")(output_3)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Đường dẫn đến thư mục chứa tất cả các thư mục con (tương ứng với các nhãn)
train_directory = r"data\extra_data_train"
validation_directory = r"data\extra_data_val"
test_directory = r"data\extra_data_test"

name_model = 'cvt_model_final_v56.keras'
input_shape = (160, 160, 3)
image_size = (160, 160)
batch_size=32
epochs=250

num_labels = 351

# Create the CvT model
model = create_cvt_model(input_shape, num_labels)

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

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Định nghĩa EarlyStopping callback với patience = 2
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping])

# Lưu mô hình sau khi huấn luyện
model.save(name_model)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
# Plot training history - Loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show() 

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = model.evaluate(test_dataset)

# Lấy các nhãn thật (true labels) và các dự đoán (predictions) từ tập test
true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Chuyển đổi các nhãn thật từ dạng one-hot sang nhãn số nguyên
true_labels = np.argmax(true_labels, axis=1)

# Tính các metrics khác như F1 score, Precision và Recall
f1 = f1_score(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

# In kết quả
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# In bảng phân loại chi tiết
print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels))