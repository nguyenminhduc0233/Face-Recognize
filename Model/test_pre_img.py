import os
import cv2
from test9 import Test9

def process_images_in_subfolders(parent_folder, dest_folder):
    # Tạo một đối tượng của lớp Test9
    test9 = Test9()
    
    # Duyệt qua từng thư mục và tệp trong thư mục cha
    for root, dirs, files in os.walk(parent_folder):
        for filename in files:
            # Kiểm tra xem tệp có phải là ảnh không
            if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                # Tạo đường dẫn đầy đủ đến tệp ảnh
                image_path = os.path.join(root, filename)
                
                # Xử lý ảnh bằng lớp Test9
                processed_face, x, y, w, h = test9.detect_face(image_path)
                
                # Xác định đường dẫn đích cho tệp đích
                # Thay thế 'data' bằng 'data_pre' trong đường dẫn
                relative_path = os.path.relpath(image_path, parent_folder)
                des_path = os.path.join(dest_folder, relative_path)
                
                # Tạo thư mục nếu không tồn tại
                os.makedirs(os.path.dirname(des_path), exist_ok=True)
                
                # Lưu ảnh xử lý vào thư mục đích
                if processed_face is not None:
                    cv2.imwrite(des_path, processed_face)
                    print(f"Đã lưu ảnh xử lý vào {des_path}")

# Thư mục cha chứa tất cả các thư mục và tệp ảnh
parent_folder = r"data_pre"

# Thư mục đích để lưu ảnh xử lý, thay thế 'data' bằng 'data_pre'
dest_folder = parent_folder.replace("data_pre", "data")

# Xử lý các ảnh trong các thư mục con của thư mục cha và lưu vào thư mục đích
process_images_in_subfolders(parent_folder, dest_folder)