import cv2
from mtcnn import MTCNN

class Test9:

    def detect_face_with_mtcnn(self, image):
        # Khởi tạo MTCNN detector
        detector = MTCNN()
        # Phát hiện khuôn mặt
        faces = detector.detect_faces(image)
        # Trả về True nếu phát hiện và False nếu không phát hiện
        return len(faces) > 0
    
    def detect_face_with_opencv(self, image):
        # Chuyển đổi ảnh sang thang độ xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Khởi tạo bộ phát hiện khuôn mặt với Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Trả về True nếu phát hiện và False nếu không phát hiện
        return len(faces) > 0

    def preprocess_image(self, image_path):
        # Đọc ảnh
        image = cv2.imread(image_path)
        
        # Chuyển hóa ảnh xám
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện và cắt mặt bằng MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image)
        if len(faces) > 0:
            # Lấy tọa độ của khuôn mặt đầu tiên
            x, y, w, h = faces[0]['box']
            x_mtcnn, y_mtcnn, w_mtcnn, h_mtcnn = x, y, w, h
            # Cắt ảnh theo khuôn mặt
            face_image_mtcnn = image[y:y+h, x:x+w]
            # Resize ảnh khuôn mặt về kích thước 160x160
            face_image_mtcnn_resized = cv2.resize(face_image_mtcnn, (160, 160))
        else:
            # Nếu không tìm thấy khuôn mặt, trả về None
            face_image_mtcnn_resized = None
            x_mtcnn, y_mtcnn, w_mtcnn, h_mtcnn = 0, 0, 0, 0
        
        # Phát hiện và cắt mặt bằng OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_opencv = face_cascade.detectMultiScale(image_gray, 1.3, 5)
        if len(faces_opencv) > 0:
            # Lấy tọa độ của khuôn mặt đầu tiên
            x, y, w, h = faces_opencv[0]
            x_opencv, y_opencv, w_opencv, h_opencv = x, y, w, h
            # Cắt ảnh theo khuôn mặt
            face_image_opencv = image[y:y+h, x:x+w]
            
            # Resize ảnh khuôn mặt về kích thước 160x160
            face_image_opencv_resized = cv2.resize(face_image_opencv, (160, 160))
        else:
            # Nếu không tìm thấy khuôn mặt, trả về None
            face_image_opencv_resized = None
            x_opencv, y_opencv, w_opencv, h_opencv = 0, 0, 0, 0
        
        # Resize ảnh gốc về kích thước 160x160
        image_resized = cv2.resize(image, (160, 160))
        
        return image_resized, face_image_mtcnn_resized, face_image_opencv_resized, x_mtcnn, y_mtcnn, w_mtcnn, h_mtcnn, x_opencv, y_opencv, w_opencv, h_opencv
    
    def detect_face(self, image_path):

        processed_image, processed_face_mtcnn, processed_face_opencv, x_mtcnn, y_mtcnn, w_mtcnn, h_mtcnn, x_opencv, y_opencv, w_opencv, h_opencv = self.preprocess_image(image_path)

        detected_with_mtcnn = False
        # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh đã cắt bằng OpenCV
        if processed_face_opencv is not None:
            detected_with_mtcnn = self.detect_face_with_mtcnn(processed_face_opencv)
            if not detected_with_mtcnn:
                detected_with_mtcnn = self.detect_face_with_opencv(processed_face_opencv)
            # In ra kết quả
            print("MTCNN detected face in OpenCV cropped image:", detected_with_mtcnn)
        else:
            print("Error: OpenCV cropped image is not valid.")

        print(detected_with_mtcnn)

        detected_with_opencv = False
        # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh đã cắt bằng OpenCV
        if processed_face_mtcnn is not None:
            detected_with_opencv = self.detect_face_with_opencv(processed_face_mtcnn)
            if not detected_with_opencv:
                detected_with_opencv = self.detect_face_with_mtcnn(processed_face_mtcnn)
            # In ra kết quả
            print("OpenCV detected face in MTCNN cropped image:", detected_with_opencv)
        else:
            print("Error: MTCNN cropped image is not valid.")

        print(detected_with_opencv)

        if detected_with_mtcnn:
            x, y, w, h = x_opencv, y_opencv, w_opencv, h_opencv
            print("Trả về OpenCV")
            return processed_face_opencv, x, y, w, h
        elif detected_with_opencv:
            x, y, w, h = x_mtcnn, y_mtcnn, w_mtcnn, h_mtcnn
            print("Trả về MTCNN")
            return processed_face_mtcnn, x, y, w, h
        else:
            return None, 0, 0, 0, 0