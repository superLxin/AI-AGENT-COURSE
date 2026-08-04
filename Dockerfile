# 第 9 讲：Docker 部署配置
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动 FastAPI
CMD ["uvicorn", "lecture-09.09_api:app", "--host", "0.0.0.0", "--port", "8000"]
