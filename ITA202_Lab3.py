# ITA202 - Lab 3: Hoi quy tuyen tinh

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Bài 1 – Xây dựng mô hình hồi quy tuyến tính


rng = np.random.default_rng(42)
X1 = np.linspace(0, 10, 100).reshape(-1, 1)
noise = rng.normal(loc=0, scale=1.5, size=100)
y1 = 2 * X1.ravel() + 1 + noise

model1 = LinearRegression()
model1.fit(X1, y1)

print('BÀI 1 - KẾT QUẢ')
print(f'Coefficient (hệ số góc): {model1.coef_[0]:.4f}')
print(f'Intercept (hệ số chặn): {model1.intercept_:.4f}')
print(f'Phương trình dự đoán: y = {model1.coef_[0]:.4f} * X + {model1.intercept_:.4f}')


# Bài 2 – Đánh giá mô hình với dữ liệu Diabetes


diabetes = load_diabetes()
X2 = diabetes.data[:, [2]]  # Chỉ lấy 1 đặc trưng: BMI (cột 2)
y2 = diabetes.target

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mae2 = mean_absolute_error(y2_test, y2_pred)
mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print('BÀI 2 - ĐÁNH GIÁ TRÊN TẬP TEST')
print(f'Số mẫu train: {len(X2_train)}, test: {len(X2_test)}')
print(f'MAE: {mae2:.4f}')
print(f'MSE: {mse2:.4f}')
print(f'R-squared (R²): {r2_2:.4f}')

# MAE: độ lệch tuyệt đối trung bình giữa dự đoán và thực tế; càng thấp càng tốt.
# MSE: trung bình bình phương sai số; phạt mạnh các dự đoán sai lệch lớn; càng thấp càng tốt.
# R²: mức độ mô hình giải thích biến thiên của y trên tập test; càng gần 1 càng tốt.
# R² = 0 nghĩa là tương đương dự đoán bằng giá trị trung bình test; R² có thể âm.


# Bài 3 – Trực quan hóa dữ liệu và đường hồi quy
# Tận dụng dữ liệu và mô hình Diabetes ở Bài 2. Các điểm là dữ liệu **test thực tế**; đường thẳng là dự đoán của mô hình.

x_line = np.linspace(X2.min(), X2.max(), 200).reshape(-1, 1)
y_line = model2.predict(x_line)

plt.figure(figsize=(9, 5))
plt.scatter(X2_test.ravel(), y2_test, alpha=0.7, label='Dữ liệu thực tế (test)')
plt.plot(x_line.ravel(), y_line, linewidth=2, label='Đường hồi quy dự đoán')
plt.xlabel('BMI (đã chuẩn hóa trong bộ dữ liệu Diabetes)')
plt.ylabel('Chỉ số tiến triển bệnh sau 1 năm')
plt.title('Bài 3: Linear Regression trên bộ dữ liệu Diabetes')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# Bài 4 – Quy trình Machine Learning hoàn chỉnh
# **Load → Preprocess (Scaling) → Train → Evaluate (MSE) → Visualize.**
# 
# Ưu tiên dữ liệu `california_housing` theo yêu cầu. Bộ này có thể cần tải về khi chạy lần đầu. **Nếu không có mạng**, ví dụ tự chuyển sang `diabetes` (đề cho phép tập dữ liệu hồi quy tương tự có sẵn trong sklearn), và có thông báo rõ nguồn dữ liệu.
# 
# Dùng `Pipeline` để đảm bảo chỉ học tham số scaling trên tập train, tránh rò rỉ dữ liệu từ test.

try:
    housing = fetch_california_housing()
    X4, y4 = housing.data, housing.target
    dataset4_name = 'California Housing'
except (OSError, ConnectionError, TimeoutError, ImportError) as error:
    print('Không tải được California Housing; chuyển sang Diabetes có sẵn.')
    print('Lý do:', str(error)[:180])
    fallback = load_diabetes()
    X4, y4 = fallback.data, fallback.target
    dataset4_name = 'Diabetes (dữ liệu thay thế)'

print('BÀI 4 - DATASET:', dataset4_name)
print('Kích thước X:', X4.shape, '| Kích thước y:', y4.shape)

# 1. Chia tập train/test trước khi scaling để tránh data leakage.
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=42
)

# 2. Scaling trên train, 3. Train Linear Regression.
model4 = make_pipeline(StandardScaler(), LinearRegression())
model4.fit(X4_train, y4_train)

# 4. Evaluate bằng MSE trên tập test.
y4_pred = model4.predict(X4_test)
mse4 = mean_squared_error(y4_test, y4_pred)
print(f'Train: {len(X4_train)} mẫu | Test: {len(X4_test)} mẫu')
print(f'MSE trên test: {mse4:.4f}')

# 5. Visualize: so sánh giá trị thực tế và giá trị dự đoán của 100 mẫu test đầu.
n_show = min(100, len(y4_test))
plt.figure(figsize=(7, 6))
plt.scatter(y4_test[:n_show], y4_pred[:n_show], alpha=0.65)
line_min = min(y4_test[:n_show].min(), y4_pred[:n_show].min())
line_max = max(y4_test[:n_show].max(), y4_pred[:n_show].max())
plt.plot([line_min, line_max], [line_min, line_max], '--', label='Dự đoán hoàn hảo (y = x)')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title(f'Bài 4: Thực tế và dự đoán - {dataset4_name}')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Sai số càng nhỏ thì mô hình càng tốt; điểm càng gần đường y=x thì dự đoán càng chính xác.


# Bài 5 – Giảng viên giao thêm
# Trong file đề bài chỉ ghi **“Giảng viên giao thêm”**, chưa cung cấp nhiệm vụ cụ thể. Bổ sung khi có hướng dẫn từ giảng viên.

