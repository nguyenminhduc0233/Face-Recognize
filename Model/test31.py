import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Test31:
    # Hàm tải mô hình và dự đoán ảnh
    def load_model_and_predict_images(self, model_path, image_arrays):
        # Tải mô hình đã được huấn luyện
        model = tf.keras.models.load_model(model_path)

        predictions = {}

        # Duyệt qua từng ảnh và dự đoán lớp
        for index, img_array in enumerate(image_arrays):
            img_array = np.expand_dims(img_array, axis=0)

            # Dự đoán lớp của ảnh
            predictions[index] = model.predict(img_array)[0]

        return predictions

    # Hàm lấy vị trí index của giá trị lớn nhất trong khoảng nhất định
    def get_index_in_range(self, arr, lower_bound, upper_bound):
        max_value = np.max(arr)
        if max_value >= lower_bound and max_value <= upper_bound:
            max_indices = np.where(arr == max_value)[0]  # Tìm tất cả các vị trí có giá trị bằng max_value
            return max_indices[0], max_value  # Chọn vị trí đầu tiên nếu có nhiều vị trí có cùng giá trị max_value
        else:
            return -1, max_value
        
    def transform_array(self, num):

        # Tạo một mảng từ 1 đến 351
        mang = list(range(1, 352))

        # Chuyển đổi mỗi số trong mảng thành chuỗi
        mang_str = list(map(str, mang))

        # Sắp xếp mảng theo thứ tự Unicode
        mang_unicode = sorted(mang_str)

        # Chuyển đổi từ chuỗi sang số
        arr1 = list(map(int, mang_unicode))

        arr2 = list(range(1, 352))  # arr2 từ 1 đến 351

        # Tìm vị trí index của giá trị num trong sorted_arr1
        index = arr2.index(num)

        # Lấy giá trị tương ứng từ sorted_arr2
        transformed_value = arr1[index]

        return transformed_value

    def prediction_face(self, model_path, image_arrays, lower_bound, upper_bound):
        
        # Tải và dự đoán các ảnh
        predictions = self.load_model_and_predict_images(model_path, image_arrays)
        predicted_classes = []  # Danh sách lưu trữ các dự đoán
        scores = [] #Danh sách lưu trữ điểm dự đoán

        # Đếm số lượng ảnh nhận diện đúng và in ra các ảnh nhận diện sai cùng với label thật sự của chúng
        for index, prediction in predictions.items():
            # Lấy vị trí index của giá trị lớn nhất trong khoảng nhất định
            predicted_class_index, max_value = self.get_index_in_range(prediction, lower_bound, upper_bound)
            if predicted_class_index is not None:
                # Lấy điểm dự đoán cụ thể của ảnh trong khoảng thời gian cụ thể
                # specific_prediction = prediction[predicted_class_index]
                # Tăng giá trị của nhãn dự đoán lên 1 để chuyển đổi từ 0 đến n-1 thành từ 1 đến n
                predicted_class_index += 1
                if predicted_class_index < 1:
                    predicted_class_index = -1
                else:
                    predicted_class_index = self.transform_array(predicted_class_index)

                predicted_classes.append(predicted_class_index)  # Thêm dự đoán vào danh sách
                scores.append(max_value)  # Thêm điểm dự đoán vào danh sách

                # # Hiển thị ảnh và nhãn dự đoán
                # plt.figure()
                # plt.imshow(image_arrays[index])
                # plt.title(f"Ảnh số {index}: Ảnh thuộc lớp {predicted_class_index}, Điểm dự đoán: {specific_prediction}")
                # plt.axis('off')
                # plt.show()

            else:
                print(f"Ảnh số {index}: Không có dự đoán trong khoảng")
                predicted_classes.append(-1)                
        
        return predicted_classes, scores