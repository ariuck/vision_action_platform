# Django 웹 서버와 Celery Worker가 같은 이미지를 쓰고, compose.yaml에서 실행 명령만 다르게 준다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --worker-class gthread --threads 8 --timeout 120"]
# gthread: 카메라 미리보기 스트림이 연결을 오래 붙잡아도 다른 요청(상태 조회 등)을 계속 처리하기 위해
