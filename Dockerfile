# Chọn base image. Python 3.9 hoặc 3.10 thường là lựa chọn tốt.
# Dùng slim để giảm kích thước image.
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy requirements.txt và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn ứng dụng của bạn vào container
COPY . .

# Đặt biến môi trường PORT mà Gradio/Streamlit/Flask/FastAPI cần
# Mặc định Hugging Face Spaces expose cổng 7860 cho Gradio/Streamlit
# Nhưng đối với API Docker, bạn có thể cần expose cổng riêng của Flask
# Hugging Face Spaces sẽ tự động expose cổng 7860 nếu có server chạy trên đó.
# Tuy nhiên, nếu bạn muốn một API thuần túy, có thể dùng cổng khác.
# Với Flask/Gunicorn, bạn thường bind đến 0.0.0.0:$PORT và Spaces sẽ handle.
# Mặc định của Spaces là 7860 nếu bạn dùng Gradio/Streamlit, nhưng với Docker API,
# cổng có thể là 80, 8000, 5000, tùy cách bạn cấu hình Gunicorn/Uvicorn.
# Tốt nhất là sử dụng PORT được cung cấp bởi môi trường.

# Expose cổng mà ứng dụng của bạn sẽ lắng nghe (cổng mà gunicorn/uvicorn bind vào)
# Nếu bạn dùng gunicorn và bind đến 0.0.0.0:$PORT, thì không cần EXPOSE cụ thể
# vì Spaces sẽ tự động ánh xạ. Nhưng tốt nhất là biết ứng dụng của bạn bind cổng nào.
# Trong trường hợp của bạn, Flask chạy trên 5000 cục bộ, nhưng trên Spaces nó sẽ là $PORT.
# Vì thế, lệnh START sẽ sử dụng $PORT.

# Lệnh chạy ứng dụng khi container khởi động
# Đảm bảo bạn đã thêm gunicorn vào requirements.txt
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
# Hoặc nếu bạn muốn cổng linh hoạt theo biến môi trường PORT (khuyến nghị):
# CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$(PORT)"]
# Lưu ý: $PORT trên Hugging Face Spaces thường là 7860 cho các demo.
# Hãy thử 7860 trước, nếu không được thì mới xem xét biến PORT.
# Với Docker, họ sẽ tự động map cổng, bạn chỉ cần lắng nghe một cổng bên trong container.
# 7860 là cổng mặc định cho Gradio/Streamlit, nên dùng nó nếu bạn muốn demo trực tiếp.
# Nếu chỉ là API thuần túy, cổng 8000 hoặc 5000 cũng được.
# Để đơn giản, cứ dùng 7860 cho lần đầu, vì nó là cổng mặc định của Spaces.