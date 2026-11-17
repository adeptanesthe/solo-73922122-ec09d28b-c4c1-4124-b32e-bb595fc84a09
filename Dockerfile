# CPU 版 PyTorch + FastAPI 运行镜像
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# 仅安装 CPU 版 torch（通过 PyTorch 官方 CPU wheel 源）
COPY requirements.txt /srv/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
       -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8000

# 容器内默认线程数收敛，避免 CPU 环境过度订阅
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
