import tensorflow as tf
from keras import models, Input, layers
import matplotlib.pyplot as plt
from keras import callbacks
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Define the CNN model
def create_cnn_model(input_shape, num_labels):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_labels, activation='softmax'))

    return model

# Đường dẫn đến thư mục chứa tất cả các thư mục con (tương ứng với các nhãn)

name_model = 'cnn_model_final_v15.keras'
input_shape = (160, 160, 3)
image_size = (160, 160)
batch_size=32
epochs=300

num_labels = 351

# Create the CNN model
model = create_cnn_model(input_shape, num_labels)

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

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Định nghĩa EarlyStopping callback với patience = 3
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