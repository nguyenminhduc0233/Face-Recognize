import React, {useEffect} from 'react';
import { View, StyleSheet, Image, Text, ScrollView, TouchableOpacity, Linking } from 'react-native';
import Svg, { Rect } from 'react-native-svg';
import { Text as SvgText } from 'react-native-svg';

const Result = ({ route }) => {
  // const { imageUri, x, y, w, h } = route.params;
  const { imageUri, infoData, locationData } = route.params;

  const [imageWidth, setImageWidth] = React.useState(0); // Chiều rộng thật sự của ảnh
  const [imageHeight, setImageHeight] = React.useState(0); // Chiều dài thật sự của ảnh
  const [locationUpdate, setLocationUpdate] = React.useState([]);
  const [effectRunCount, setEffectRunCount] = React.useState(0);

  React.useEffect(() => {
    try {
      Image.getSize(imageUri, (width, height) => {
        setImageWidth(width);
        setImageHeight(height);
        console.log('Loaded image width:', width);
        console.log('Loaded image height:', height);
      });
    } catch (error) {
      console.error("Error fetching image size:", error);
    }
  }, [imageUri]);

  useEffect(() => {
    if (imageWidth > 0 && imageHeight > 0) {
      console.log('imageWidth: ', imageWidth);
      console.log('imageHeight: ', imageHeight);
      // ... Tính toán và cập nhật locationUpdate ở đây ...
    } else if (effectRunCount < 10) {
      // Chỉ tăng effectRunCount nếu imageWidth và imageHeight vẫn bằng 0
      setEffectRunCount(effectRunCount + 1);
    }
  }, [imageWidth, imageHeight, effectRunCount]);
  
  const isValidValue = (value) => {
    return isFinite(value) && !isNaN(value);
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollView}>
      <View style={styles.imageContainer}>
        {imageUri ? (
          <View style={{ flex: 1 }}>
            <Image
              source={{ uri: imageUri }}
              style={styles.image}
              onLoad={(event) => {
                const { width, height } = event.nativeEvent.source;
                console.log('Loaded image width:', width);
                console.log('Loaded image height:', height);
                setImageWidth(width);
                setImageHeight(height);
              }}
              onLayout={(event) => {
                const { width, height } = event.nativeEvent.layout; // Kích thước chiều dài, cao vùng hiển thị ảnh
                console.log('imageWidth: ', imageWidth);
                console.log('imageHeight: ', imageHeight);
                var scale = 0;
                if(imageWidth < imageHeight){
                  scale = height / imageHeight;
                }else{
                  scale = width / imageWidth;
                }
                let locationUpdateTemp = [];
                for(let location_item of locationData){
                  var newX = 0;
                  var newY = 0;
                  if(imageWidth < imageHeight){
                    const imgwidth = imageWidth * scale; // Kích thước chiều tộng mà ảnh hiển thị trên màn hình
                    newX = location_item[0] * scale + (width - imgwidth) / 2;
                    newY = location_item[1] * scale;
                  }else{
                    const imgheight = imageHeight * scale; // Kích thước chiều tộng mà ảnh hiển thị trên màn hình
                    newX = location_item[0] * scale;
                    newY = location_item[1] * scale + (height - imgheight) / 2;
                  }
                  const newW = location_item[2] * scale;
                  const newH = location_item[3] * scale;

                  if (isValidValue(newX) && isValidValue(newY) && isValidValue(newW) && isValidValue(newH)) {
                    const new_location = { newX, newY, newW, newH };
                    locationUpdateTemp.push(new_location);
                  }
                }
                setLocationUpdate(locationUpdateTemp);
              }}
            />
            <Svg height="100%" width="100%" style={styles.svg}>
              {locationUpdate.length > 0 && locationUpdate.map((rect, index) => (
                <React.Fragment key={index}>
                  <Rect
                    x={rect.newX}
                    y={rect.newY}
                    width={rect.newW}
                    height={rect.newH}
                    fill="transparent" // Đặt màu nền là trong suốt
                    stroke="green" // Đặt màu viền là xanh
                  />
                  <SvgText
                    fill="red" // Màu chữ là đỏ
                    fontSize="20"
                    fontWeight="bold"
                    x={rect.newX + rect.newW * 0.9} // Đặt chữ ở giữa theo chiều ngang
                    y={rect.newY + rect.newH / 1.1} // Đặt chữ ở giữa theo chiều dọc
                    textAnchor="middle" // Căn giữa chữ
                  >
                    {index + 1}
                  </SvgText>
                </React.Fragment>
              ))}
            </Svg>
          </View>
        ) : (
          <Text>Đang tải ảnh...</Text>
        )}
      </View>
      <View style={{ ...styles.infoContainer, flex: 1 }}>
      <View>
        {infoData.length === 0 ? (
          <Text>Không phát hiện được khuôn mặt nào</Text>
        ) : (
          infoData.map((face, index) => (
            <React.Fragment key={index}>
              <Text>
                <Text style={styles.boldText}>Khuôn mặt số:</Text> {index + 1}
              </Text>
              <Text>
                <Text style={styles.boldText}>Tên:</Text> {face.name}
              </Text>
              <Text>
                <Text style={styles.boldText}>Thông tin:</Text> {face.profile}
              </Text>
              <TouchableOpacity onPress={() => Linking.openURL(face.link)}>
                <Text>
                  <Text style={styles.boldText}>Link: </Text>
                  <Text style={styles.linkText}>{face.link}</Text>
                </Text>
              </TouchableOpacity>
            </React.Fragment>
          ))
        )}
      </View>
      </View>
    </ScrollView>    
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flexGrow: 1,
  },
  imageContainer: {
    alignItems: 'left',
    width: '100%',
    marginBottom: 20,
  },
  image: {
    width: '100%',
    height: 300,
    resizeMode: 'contain',
    position: 'relative',
  },
  svg: {
    position: 'absolute',
  },
  infoContainer: {
    paddingHorizontal: 20,
  },
  boldText: {
    fontWeight: 'bold',
  },
  linkText: {
    color: '#0000ee',
  },
});

export default Result;
