from test31 import Test31
from test9 import Test9
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from fileprocessor import FileProcessor

def get_nd_by_id(json_file, target_id):
    # Đọc dữ liệu từ file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Tìm kiếm trong dữ liệu JSON
    for id_nd_pair in data:
        if id_nd_pair['id'] == target_id:
            return id_nd_pair['nd']
    
    # Trả về None nếu không tìm thấy id tương ứng
    return None

def process_faces_in_image(image_path):
    # Khởi tạo một đối tượng từ lớp Test9
    test_object = Test9()

    # Đọc ảnh gốc
    original_image = cv2.imread(image_path)
    original_image1 = cv2.imread(image_path)
    # Chuyển đổi từ BGR sang RGB
    original_image1 = cv2.cvtColor(original_image1, cv2.COLOR_BGR2RGB)

    # Tạo thư mục phụ để lưu các ảnh sao
    temp_folder = "temp_images"
    os.makedirs(temp_folder, exist_ok=True)  # Tạo thư mục phụ nếu chưa tồn tại

    # Tạo bản sao của ảnh gốc và lưu vào thư mục phụ
    temp_image_path = os.path.join(temp_folder, "temp_image.png")
    cv2.imwrite(temp_image_path, original_image)

    temp_image_path1 = os.path.join(temp_folder, "temp_image1.png")
    cv2.imwrite(temp_image_path1, original_image1)

    # Đọc ảnh từ thư mục phụ để thực hiện xử lý
    temp_image = cv2.imread(temp_image_path)
    temp_image1 = cv2.imread(temp_image_path1)

    # Danh sách các mảng numpy đại diện cho các khuôn mặt đã được xử lý
    processed_faces = []
    cropped_faces = []
    location_faces = []

    # Lặp lại việc xử lý cho đến khi không còn phát hiện được khuôn mặt nào
    while True:
        # Gọi hàm detect_face để nhận được mảng numpy đại diện cho khuôn mặt
        face_array, x, y, w, h = test_object.detect_face(temp_image_path)

        # Kiểm tra xem hàm detect_face có trả về mảng numpy không
        if face_array is not None:
            # Bôi đen phần khuôn mặt trên ảnh sao
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Vẽ một hình chữ nhật màu đen để bôi đen khuôn mặt

            # Lưu ảnh sau khi đã bị bôi đen
            cv2.imwrite(temp_image_path, temp_image)

            location_faces.append((x, y, w, h))
            processed_faces.append(face_array)

            # Lấy phần ảnh tương ứng với khuôn mặt từ ảnh gốc và lưu vào mảng cropped_faces
            cropped_face = temp_image1[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
        else:
            # Nếu không còn phát hiện được khuôn mặt nào, thoát khỏi vòng lặp
            break

    # Xóa ảnh sao và thư mục phụ
    os.remove(temp_image_path)
    os.remove(temp_image_path1)
    # os.rmdir(temp_folder)

    return processed_faces, cropped_faces, location_faces


# # Định nghĩa khoảng giá trị cho dự đoán
lower_bound = 0.3
upper_bound = 1.0

cvt_model_path = r"model_vit\vit_model_final_v9.keras"
image_folder = r"data\test"
test31 = Test31()
fileprocessor = FileProcessor()
for image_path in fileprocessor.print_path_files(image_folder):
    print(image_path)
    arr_num, cropped_faces, location_faces = process_faces_in_image(image_path)
    if len(arr_num) != 0:

        pre_class, predictions = test31.prediction_face(cvt_model_path, arr_num, lower_bound, upper_bound)

        # Đọc ảnh từ đường dẫn
        img = Image.open(image_path)
                
        # Tạo plot và hiển thị ảnh
        fig, ax = plt.subplots()
        ax.imshow(img)
                
        # Vẽ các hình chữ nhật và đánh số
        for index, (x, y, w, h) in enumerate(location_faces):
            # Tạo hình chữ nhật bao quanh khuôn mặt với các kích thước x, y, w, h
            print(w)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
            
            # Thêm hình chữ nhật vào plot
            ax.add_patch(rect)
            
            # Đánh số thứ tự trên hình chữ nhật
            ax.text(x, y - 10, str(index + 1), color='green', fontsize=12, ha='center')

        # Hiển thị ảnh với các hình chữ nhật bao quanh khuôn mặt và đánh số
        plt.show()

        for i in range(len(arr_num)):
            image = cropped_faces[i]
            integer = pre_class[i]
            score = predictions[i]
            
            plt.imshow(image)
            
            # Đặt tiêu đề cho hình ảnh
            if integer == -1:
                plt.title("Không xác định")
            else:
                json_file = 'output.json'
                title = get_nd_by_id(json_file, integer)
                title_with_number = f"{i+1}: {title} - {score}"
                if title is not None:
                    plt.title(title_with_number)
                else:
                    plt.title("???")
            
            plt.axis('off')  # Tắt trục
            plt.show()
    else:
        print("Không có ảnh khuôn mặt được phát hiện")

print("Hello World")