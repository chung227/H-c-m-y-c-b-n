
# ============================================================
# BAI 1: LAM QUEN VOI NUMPY VA PANDAS
# ============================================================

import numpy as np
import pandas as pd

# 1. Tao mang Numpy 1 chieu chua cac so tu 1 den 10
numbers = np.arange(1, 11)
print("Mang Numpy tu 1 den 10:")
print(numbers)

# 2. Tao DataFrame gom thong tin cua 5 sinh vien
students = {
    "Ho ten": ["Nguyen Van An", "Tran Thi Binh", "Le Minh Chau",
               "Pham Quoc Dung", "Hoang Thu Ha"],
    "Tuoi": [20, 21, 19, 22, 20],
    "Diem toan": [8.5, 7.0, 9.0, 6.5, 8.0]
}

df = pd.DataFrame(students)

print("\nThong tin 5 sinh vien:")
print(df)

# 3. Hien thi thong tin tong quan cua DataFrame
print("\nThong tin tong quan:")
df.info()

# Thong ke mo ta
print("\nThong ke mo ta:")
print(df.describe())

# 4. Tinh diem trung binh mon Toan
average_math = df["Diem toan"].mean()
print("\nDiem trung binh mon Toan:", round(average_math, 2))


# ============================================================
# BAI 2: PHAN LOAI BAI TOAN MACHINE LEARNING
# ============================================================

# 1. Du doan gia nha dua tren dien tich:
#    Supervised Learning (Hoc co giam sat)
#    Vi du: du lieu huan luyen co dien tich va gia nha da biet.
#    Day la bai toan hoi quy (Regression).

# 2. Gom nhom khach hang dua tren hanh vi mua sam:
#    Unsupervised Learning (Hoc khong giam sat)
#    Vi du: du lieu khong co nhan nhom san, mo hinh tu tim cac nhom.
#    Day la bai toan phan cum (Clustering).

# 3. Phan loai email la Spam hay khong:
#    Supervised Learning (Hoc co giam sat)
#    Vi du: email trong tap huan luyen da duoc gan nhan Spam/Not Spam.
#    Day la bai toan phan loai (Classification).

# 4. Nhan dien chu viet tay tu hinh anh:
#    Supervised Learning (Hoc co giam sat)
#    Vi du: hinh anh chu viet tay duoc gan nhan tuong ung (0-9).
#    Day la bai toan phan loai (Classification).

print("\nBai 2: Da ghi chu phan loai 4 vi du trong comment.")


# ============================================================
# BAI 3: CHAY THU MO HINH DON GIAN VOI SCIKIT-LEARN
# ============================================================

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# 1. Load du lieu Iris
iris = load_iris()
X = iris.data       # 4 dac trung cua moi mau hoa
y = iris.target     # Nhan lop hoa: 0, 1, 2

print("\nBai 3 - Kich thuoc du lieu Iris:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 2. Khoi tao mo hinh Linear Regression
model = LinearRegression()

# 3. Fit model (huan luyen)
model.fit(X, y)

# 4. Predict (du doan)
predictions = model.predict(X)

print("\n5 gia tri du doan dau tien:")
print(predictions[:5])

print("\n5 nhan thuc te dau tien:")
print(y[:5])

# Luu y: Iris la bai toan phan loai, nhung theo yeu cau Lab,
# ta thu LinearRegression de quan sat cach model fit va predict.
# Gia tri du doan cua Linear Regression co the la so thap phan,
# khong phai nhan lop 0, 1, 2 truc tiep.


# ============================================================
# BAI 4: QUY TRINH ML - THU THAP VA KIEM TRA DU LIEU
# ============================================================

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Tai tap du lieu Digits
digits = load_digits()

X_digits = digits.data
y_digits = digits.target

# 2. In kich thuoc (shape) cua du lieu va nhan
print("\nBai 4 - Du lieu Digits:")
print("X_digits shape:", X_digits.shape)
print("y_digits shape:", y_digits.shape)

# 3. Hien thi mau du lieu dau tien duoi dang hinh anh
first_image = digits.images[0]
first_label = y_digits[0]

plt.figure(figsize=(4, 4))
plt.imshow(first_image, cmap="gray")
plt.title(f"Chu so trong mau dau tien: {first_label}")
plt.axis("off")
plt.show()

# In gia tri cua mau dau tien
print("\nGia tri ma tran cua mau dau tien:")
print(first_image)

print("\nNhan cua mau dau tien:", first_label)


# ============================================================
# BAI 5: GIANG VIEN GIAO THEM
# ============================================================
# De bai chua duoc cung cap trong file PDF.
# Bo sung code sau khi co yeu cau cu the tu giang vien.
