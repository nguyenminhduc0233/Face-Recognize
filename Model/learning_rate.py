import matplotlib.pyplot as plt

# Tạo mảng giá trị
array1 = [0.000081, 0.000171, 0.000206, 0.000248, 0.000433, 0.000756, 0.001097, 0.004863, 0.010235, 0.037649]
array2 = [5.833217, 5.923424, 5.957707, 5.864351, 5.885746, 5.668798, 5.944179, 5.870525, 5.662209, 5.962576]

array3 = [0.000001, 0.000013, 0.000248, 0.000298, 0.000433, 0.000521, 0.001097, 0.001592, 0.007055, 0.014850]
array4 = [5.828649, 5.779496, 5.742426, 5.893757, 5.750080, 5.798444, 5.774364, 5.872519, 5.863487, 5.509608]

array5 = [0.000118, 0.000142, 0.000206, 0.000359, 0.000521, 0.001592, 0.002310, 0.004037, 0.010235, 0.014850]
array6 = [5.775298, 5.845912, 5.810432, 5.737315, 5.840796, 5.778485, 5.597257, 5.753326, 5.650928, 5.879350]

array7 = [0.000046, 0.000142, 0.000171, 0.000298, 0.000359, 0.000628, 0.001322, 0.001592, 0.007055, 0.014850]
array8 = [5.943235, 5.910477, 6.004787, 5.843601, 5.928164, 5.919319, 6.017114, 5.942837, 6.010809, 5.904266]

array9 = [0.000039, 0.000098, 0.000142, 0.000248, 0.000298, 0.000756, 0.001918, 0.002783, 0.012328, 0.045349]
array10 = [5.842580, 5.802954, 5.858850, 5.881673, 5.808737, 5.845203, 5.832494, 5.861193, 5.752799, 5.832092]

# Vẽ đường thứ nhất
plt.scatter(array1, array2, color='blue', marker='o', label='times 1')

# Vẽ đường thứ hai
plt.scatter(array3, array4, color='red', marker='o', label='times 2')

plt.scatter(array5, array6, color='green', marker='o', label='times 3')

plt.scatter(array7, array8, color='black', marker='o', label='times 4')

plt.scatter(array9, array10, color='brown', marker='o', label='times 5')

# Thêm tiêu đề và nhãn trục
plt.title('Biểu đồ quan hệ giữa learning rate và loss')
plt.xlabel('learning rate')
plt.ylabel('loss')

# Hiển thị legend (chú thích)
plt.legend()

# Hiển thị biểu đồ
plt.show()