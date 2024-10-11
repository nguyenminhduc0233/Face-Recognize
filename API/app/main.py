from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from app.model.predictor import ModelPredictor
import io
import os
import json

app = FastAPI()

# Khởi tạo ModelPredictor với đường dẫn đến mô hình và kích thước đích
try:
    with open('parameter.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
        {"error": "File not found"}

image_size = data[0]['image_size']
model_predictor = ModelPredictor(model_path="model.keras", target_size=(image_size, image_size))

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# 2. predict_image(file: UploadFile)
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Đọc nội dung file và mở bằng Pillow
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Lưu ảnh tạm thời với đuôi tương ứng của tệp ảnh gốc
        temp_image_path = f"temp_image_main.{file.filename.split('.')[-1]}"
        image.save(temp_image_path)
        
        # Sử dụng ModelPredictor hoặc Test31 để dự đoán
        pre_class, predictions, location_faces = model_predictor.predict(temp_image_path)
        
        # Xóa ảnh tạm sau khi đã sử dụng xong
        os.remove(temp_image_path)
        
        pre_class_json = convert_float32_to_json_serializable(pre_class)
        predictions_json = convert_float32_to_json_serializable(predictions)
        # Chuyển đổi các phần tử trong location_faces thành tuple trước khi trả về
        location_faces_tuples = [(x, y, w, h) for x, y, w, h in location_faces]
        return {"pre_class": pre_class_json, "predictions": predictions_json, "location_faces": location_faces_tuples}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def convert_float32_to_json_serializable(arr):
    arr_float64 = []
    for x in arr:
        # Kiểm tra xem phần tử có phải là tuple không
        if isinstance(x, tuple):
            # Nếu là tuple, in ra và bỏ qua
            print(f"Tuple found: {x}")
            continue
        # Chuyển đổi từng phần tử trong danh sách từ float32 sang float64
        arr_float64.append(float(x))
    return arr_float64

# 6. get_info_by_id(target_id: int)
@app.get("/info/{target_id}")
def get_info_by_id(target_id: int):
    try:
        with open('singers_info.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "File not found"}

    for id_nd_pair in data:
        if id_nd_pair['id'] == target_id:
            return {"name":id_nd_pair['name'], "profile":id_nd_pair['profile'], "link":id_nd_pair['link']}

    # Trả về thông báo nếu không tìm thấy singer
    return {"name":"Không xác định", "profile":"Không xác định", "link":""}