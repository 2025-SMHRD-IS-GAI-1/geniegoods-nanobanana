# 빌드 스테이지
FROM python:3.12-slim as builder

WORKDIR /app

# 빌드에 필요한 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 빌드 도구 설치 (시스템 레벨)
RUN pip install --upgrade pip && \
    pip install setuptools wheel build

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