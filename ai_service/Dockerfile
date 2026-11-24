# FastAPI 추론 서버. 컨테이너에는 GPU를 붙이지 않으므로 CPU용 torch를 먼저 설치한다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app

# opencv·ultralytics 실행에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# 모델 파일(models/)은 Git에 없으므로 이미지에 굽지 않고 compose에서 읽기 전용으로 연결한다.
COPY config ./config
COPY src ./src
COPY services ./services
COPY routers ./routers
COPY tools ./tools
COPY main.py .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
