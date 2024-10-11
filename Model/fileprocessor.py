import os

class FileProcessor:
    def print_path_files(self, folder_path):
        my_array = []
        # Lấy danh sách tất cả các tệp và thư mục con trong thư mục folder_path
        items = os.listdir(folder_path)
        
        # Duyệt qua từng phần tử trong danh sách
        for item in items:
            # Tạo đường dẫn đầy đủ đến phần tử hiện tại
            item_path = os.path.join(folder_path, item)
            # Kiểm tra xem phần tử là tệp hay thư mục
            if os.path.isfile(item_path):
                my_array.append(item_path)
            elif os.path.isdir(item_path):
                # Nếu là thư mục, lặp lại quá trình cho thư mục này
                sub_files = self.print_path_files(item_path)
                my_array.extend(sub_files)
        return my_array
    
    def create_output_path(self, input_path):
        try:
            # Tạo output path mới với phần mở rộng là .png và thay đổi "data" thành "data_pre"
            output_path = os.path.splitext(input_path)[0].replace("data_search", "data") + '.png'
            
            # In ra thông báo thành công trước khi thực hiện bất kỳ thao tác nào khác
            print("Tạo output thành công!")

            return output_path
        except FileNotFoundError:
            print("Không tìm thấy tệp ảnh.")
        except Exception as e:
            print("Đã xảy ra lỗi:", str(e))