import os
import numpy as np
import tensorflow as tf

# Tải MNIST dataset từ TensorFlow
mnist = tf.keras.datasets.cifar100
(train_images, train_labels), _ = mnist.load_data()

# Thư mục chính để lưu trữ dữ liệu
main_folder = "CIFAR100_Dataset"

# Tạo thư mục chính nếu chưa tồn tại
if not os.path.exists(main_folder):
    os.makedirs(main_folder)

# Duyệt qua từng nhãn (0-9)
for label in range(10):
    label_folder = os.path.join(main_folder, str(label))

    # Tạo thư mục cho từng nhãn nếu chưa tồn tại
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

# Duyệt qua từng ảnh và nhãn, lưu vào các thư mục tương ứng
for idx, (image, label) in enumerate(zip(train_images, train_labels)):
    label_folder = os.path.join(main_folder, str(label))
    image_filename = f"{label}_{idx}.png"
    image_path = os.path.join(label_folder, image_filename)

    # Mở rộng tensor từ 2 chiều lên 3 chiều bằng cách thêm chiều cuối cùng (channels)
    image = np.expand_dims(image, axis=-1)
    image = tf.keras.preprocessing.image.array_to_img(image)

    # Lưu ảnh vào thư mục
    image.save(image_path)

print("Tạo cấu trúc thư mục thành công!")
