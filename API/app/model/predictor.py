import os
import cv2
from app.test9 import Test9
from app.test31 import Test31
import json

class ModelPredictor:
    def __init__(self, model_path: str, target_size: tuple):
        self.target_size = target_size
        self.model = model_path
    
    def process_faces_in_image(self, image_path: str):
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
                location_faces.append((float(x), float(y), float(w), float(h)))
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

        return processed_faces, cropped_faces, location_faces

    def predict(self, image_path: str):
        test31 = Test31()
        arr_num, cropped_faces, location_faces = self.process_faces_in_image(image_path)
        if len(arr_num) != 0:
            try:
                with open('parameter.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                return {"error": "File not found"}

            upper_bound = data[0]['upper_bound']
            lower_bound = data[0]['lower_bound']
            pre_class, predictions = test31.prediction_face(self.model, arr_num, lower_bound, upper_bound)
        else:
            pre_class = []
            predictions = []
        
        return pre_class, predictions, location_faces