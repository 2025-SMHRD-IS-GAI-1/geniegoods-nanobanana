# 빌드 스테이지
FROM python:3.12-slim as builder

WORKDIR /app

# pip 업그레이드 및 빌드 도구 설치
RUN pip install --upgrade pip setuptools wheel

# Python 패키지 설치 (사용자 디렉토리에 설치)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 실행 스테이지
FROM python:3.12-slim

WORKDIR /app

# 빌드 스테이지에서 Python 패키지 복사
COPY --from=builder /root/.local /root/.local

# 애플리케이션 코드 복사
COPY main.py .

# PATH에 로컬 패키지 추가
ENV PATH=/root/.local/bin:$PATH

# 포트 노출
EXPOSE 8001

# 애플리케이션 실행
CMD ["python", "main.py"]