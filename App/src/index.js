import React, { useState } from 'react';
import { View, TouchableOpacity, StyleSheet, Text, Image, Alert, ImageBackground } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import Spinner from 'react-native-loading-spinner-overlay';
import * as ImageManipulator from 'expo-image-manipulator';

const Index = ({ navigation }) => {
  const [imageUri, setImageUri] = useState(null);
  const [loading, setLoading] = useState(false);

  const getInfoById = async (targetId) => {
    try {
      // 5. call API
      const response = await axios.get(`http://18.143.139.1/info/${targetId}`);
      // 7. return
      return response.data;
    } catch (error) {
      console.error('Lỗi khi lấy thông tin:', error.response ? error.response.data : error.message);
      Alert.alert('Error', 'An error occurred while fetching information.');
      return null;
    }
  };

  // 1. predict
  const uploadImage = async (uri) => {
    const formData = new FormData();
    formData.append('file', {
      uri,
      type: 'image/jpeg', // or the correct type for your image
      name: 'test.jpg', // or any name you want
    });

    setLoading(true);  // Hiển thị spinner
    try {
      // 2. call API
      const response = await axios.post('http://18.143.139.1/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setLoading(false);  // Ẩn spinner
      // 4. return
      return response.data;
    } catch (error) {
      setLoading(false);  // Ẩn spinner nếu có lỗi
      console.error('Lỗi tải lên:', error.response ? error.response.data : error.message);
      Alert.alert('Error', error);
    }
  };

  const checkImageSize = (width, height) => {
    return width > 2000 || height > 2000;
  };

  const openLibrary = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permissionResult.granted === false) {
      alert('Cần cấp quyền truy cập thư viện ảnh!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.cancelled) {
      const asset = result.assets ? result.assets[0] : { uri: result.uri, type: null, width: result.width, height: result.height };
      let resizedUri = asset.uri;
  
      // Kiểm tra nếu kích thước ảnh lớn hơn 2000x2000 và thực hiện resize nếu cần
      if (checkImageSize(asset.width, asset.height)) {
        const ratio = Math.min(2000 / asset.width, 2000 / asset.height);
        const newWidth = Math.round(asset.width * ratio);
        const newHeight = Math.round(asset.height * ratio);
  
        resizedUri = await resizeImage(asset.uri, newWidth, newHeight);
      }
  
      setImageUri(resizedUri);
    }
  };

  const resizeImage = async (uri, width, height) => {
    try {
      const manipResult = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width, height } }],
        { compress: 1, format: ImageManipulator.SaveFormat.JPEG }
      );
      return manipResult.uri;
    } catch (error) {
      console.error('Error resizing image:', error);
      Alert.alert('Error', 'Could not resize the image.');
    }
  };

  const openCamera = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (permissionResult.granted === false) {
      alert('Cần cấp quyền truy cập camera!');
      return;
    }
  
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 1,
    });
  
    if (!result.cancelled) {
      const asset = result.assets ? result.assets[0] : { uri: result.uri, type: null, width: result.width, height: result.height };
      let resizedUri = asset.uri;
  
      // Kiểm tra nếu kích thước ảnh lớn hơn 2000x2000
      if (asset.width > 2000 || asset.height > 2000) {
        // Tính toán tỉ lệ mới để giữ nguyên tỉ lệ ảnh
        const ratio = Math.min(2000 / asset.width, 2000 / asset.height);
        const newWidth = Math.round(asset.width * ratio);
        const newHeight = Math.round(asset.height * ratio);
  
        // Thực hiện resize ảnh sử dụng expo-image-manipulator
        resizedUri = await resizeImage(asset.uri, newWidth, newHeight);
      }
  
      setImageUri(resizedUri);
    }
  };

  return (
    <View style={styles.container}>
      <Spinner visible={loading} textContent={'Đang phân tích...'} textStyle={styles.spinnerTextStyle} />
      <View style={styles.imageContainer}>
        <View style={styles.backgroundOverlay} />
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.image} />
        ) : (
          <Text>Chưa có ảnh được chọn</Text>
        )}
      </View>

      <View style={styles.bottomButtons}>
        <TouchableOpacity onPress={openLibrary}>
          <ImageBackground source={require('../assets/thu_vien.png')} style={[styles.libraryButton, { borderRadius: 10 }, { overflow: 'hidden' }]}>
          </ImageBackground>
          <Text style={[{ top: 10 }]}>Chọn ảnh</Text>
        </TouchableOpacity>
        <View style={styles.circle_main}>
          <TouchableOpacity
            onPress={async () => {
              if (imageUri) {
                setLoading(true);
                const result = await uploadImage(imageUri);
                setLoading(false);
                const pre_class = result.pre_class; // Lấy danh sách các lớp dự đoán
                const infoData = []; // lấy thông tin của ca sĩ
                for (let class_item of pre_class){
                  const info_singer = await getInfoById(class_item);
                  infoData.push(info_singer);
                }
                const locationData = result.location_faces; // Lấy danh sách các location khuôn mặt
                if (result) {
                  // 8. navigate()
                  navigation.navigate('Kết quả', {
                    imageUri: imageUri,
                    infoData: infoData,
                    locationData: locationData,
                  });
                }                
              } else {
                Alert.alert('Lỗi', 'Không có ảnh để phân tích');
              }
            }}
          >
            <ImageBackground source={require('../assets/49116.png')} style={[styles.magnifierButton, { borderRadius: 10 }, { overflow: 'hidden' }]}>
            </ImageBackground>
          </TouchableOpacity>
        </View>
        <TouchableOpacity onPress={openCamera}>
          <ImageBackground source={require('../assets/camera.png')} style={styles.cameraButton}>
          </ImageBackground>
          <Text style={[{ top: 10 }]}>Máy ảnh</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageContainer: {
    flex: 6,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#D2E7FC',
  },
  image: {
    width: '90%',
    height: '90%',
    resizeMode: 'contain',
  },
  backgroundOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: '#ffffff',
    borderRadius: 30,
    borderTopLeftRadius: 0,
    borderTopRightRadius: 0,
  },
  bottomButtons: {
    flex: 3,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    paddingHorizontal: 20,
    backgroundColor: '#D2E7FC',
  },
  libraryButton: {
    top: 9,
    left: 9,
    padding: 10,
    width: 30,
    height: 30,
  },
  cameraButton: {
    top: 9,
    left: 11,
    padding: 10,
    width: 30,
    height: 30,
  },
  magnifierButton: {
    top: 15,
    left: 15,
    padding: 10,
    width: 50,
    height: 50,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
  circle_main: {
    width: 80,
    height: 80,
    borderRadius: 50,
    backgroundColor: '#2d96de', //#0B85FF
  },
  spinnerTextStyle: {
    color: '#FFF',
  },
});

export default Index;
