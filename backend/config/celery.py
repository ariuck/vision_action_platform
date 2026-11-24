"""Django 프로젝트에 Celery 앱을 연결한다. 각 앱의 tasks.py를 자동으로 찾는다."""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
