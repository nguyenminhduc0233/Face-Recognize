import tensorflow as tf
from keras import models, Input, layers
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

input_shape = (160, 160, 3)
image_size = (160, 160)
batch_size=32
epochs=200

num_labels = 351

# Create the CNN model
model = create_cvt_model(input_shape, num_labels)

# Biên dịch mô hình với optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')

# Tạo dataset cho dữ liệu huấn luyện
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size)

# Hàm tìm learning rate
def lr_finder(model, dataset, min_lr=1e-7, max_lr=10, steps=100):
    lr_list = np.geomspace(min_lr, max_lr, num=steps)
    loss_list = []

    for i, lr in enumerate(lr_list):
        # Tạo optimizer mới với learning rate mới
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        
        batch_loss = []
        for batch in dataset.take(1):  # chỉ lấy một batch từ dataset
            loss = model.train_on_batch(batch[0], batch[1])
            batch_loss.append(loss)
        loss_list.append(np.mean(batch_loss))

    return lr_list, loss_list

# Tìm learning rate
lr_list, loss_list = lr_finder(model, train_dataset)

# Tìm 5 giá trị learning rate có loss thấp nhất
lowest_losses_indices = np.argsort(loss_list)[:10]
lowest_losses = [(lr_list[i], loss_list[i]) for i in lowest_losses_indices]

# Sắp xếp theo thứ tự learning rate từ nhỏ đến lớn
lowest_losses.sort()

# In ra 10 learning rate có loss thấp nhất
for lr, loss in lowest_losses:
    print(f"Learning Rate: {lr:.6f}, Loss: {loss:.6f}")

# Vẽ biểu đồ
plt.plot(lr_list, loss_list)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
