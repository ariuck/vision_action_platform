# 개발 기록

만들면서 막혔던 것과 어떻게 풀었는지 적어 둔다.

## 1. 서비스와 학습의 특징 계산이 같은지 확인

- 문제: 서비스(FastAPI)에서 특징을 다시 계산하는데, 학습 때와 조금이라도 다르면 정확도가 떨어진다.
- 확인: `ai_service/tools/check_feature_parity.py`로 같은 영상에서 서비스 특징과 학습 시퀀스 CSV를 비교했다.
- 결과: 1,350개 특징의 최대 차이 0.000002 (반올림 수준). 그대로 사용.

## 2. 업로드하자마자 작업이 실패 (`No module named 'packaging'`)

- 문제: 로컬에서 영상을 올리니 바로 FAILURE가 뜨고 에러 코드가 QUEUE_UNAVAILABLE이었다.
- 원인: backend 가상환경에 Celery 실행에 필요한 `packaging` 패키지가 빠져 있었다. 작업 등록 중 예외가 나서 대기열 실패로 처리됐다.
- 해결: `packaging==26.4`를 설치하고 `backend/requirements.txt`에 버전을 적었다.
- 결과: 같은 영상 두 개로 다시 확인

| 영상 | 예측 | Confidence | 전체 처리 시간 |
|---|---|---|---|
| S05_A007_T01 (바닥에 앉기) | A007 바닥에 앉기 | 0.72 | 20.7초 |
| S05_A016_T03 (뒤로 넘어지기) | A016 뒤로 넘어지기 | 0.81 | 13.7초 |

FastAPI에 직접 보냈을 때와 같은 값이 나와서 Django → Celery → FastAPI → DB 저장까지 흐름이 맞는 것을 확인했다. (로컬 GPU 기준, 컨테이너는 CPU라 더 느릴 수 있음)

## 3. 좁은 화면에서 표가 찌그러짐

- 문제: 브라우저 폭이 좁으면 이력 표 글자가 한 글자씩 줄바꿈됐다.
- 해결: 표를 `.table-wrap`(가로 스크롤)으로 감싸고, 800px 이하에서 상단 메뉴가 줄바꿈되게 했다.

## 4. 컨테이너에서 CSS가 안 나오는 문제 대비

- DEBUG=0이면 Django가 static 파일을 주지 않는다. gunicorn으로 띄울 것이라 `whitenoise`를 넣고 이미지 빌드 때 `collectstatic`을 실행하게 했다.
- 컨테이너로 띄운 뒤 `/static/css/app.css`가 200으로 나오는 것 확인.

## 5. Docker Compose로 5개 컨테이너 실행

- `docker compose up -d --build` 후 postgres, redis, fastapi, django, celery 모두 healthy.
- 로컬에서는 확인용으로 Celery를 바로 실행(eager)했지만, 컨테이너에서는 실제로 Redis 대기열을 거친다. 업로드하면 바로 202와 PENDING이 오고, Worker가 처리한 뒤 SUCCESS로 바뀌었다.

| 영상 | 예측 | Confidence | 전체 처리 시간 (CPU) |
|---|---|---|---|
| S05_A007_T01 | A007 바닥에 앉기 | 0.71 | 30.8초 |
| S05_A016_T03 | A016 뒤로 넘어지기 | 0.81 | 21.7초 |

- 컨테이너는 GPU 없이 CPU로 돌려서 로컬(GPU)보다 약 1.5배 느렸다. Confidence는 소수 둘째 자리에서 조금 다르지만 예측 동작은 같다.

## 6. 재시작 후 데이터 유지

- `docker compose down` → `docker compose up -d` 후에도 작업 2개, 예측 2개, 업로드 영상 경로가 그대로 남았다.
- PostgreSQL 데이터와 업로드 영상을 이름 있는 볼륨(`postgres_data`, `media_data`)에 두었기 때문이다. `down -v`를 쓰면 볼륨까지 지워지니 주의.
