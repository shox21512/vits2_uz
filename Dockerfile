# PyTorch 2.2.0 + CUDA 12.1 runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# OS paketlar: espeak-ng (phonemizer backend), audio va utils
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget ffmpeg sox espeak-ng libespeak-ng1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python kutubxonalar
# (requirements.txt ichida borlarini ham o‘rnatamiz; keyin versiya-pinlarni qo‘shamiz)
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch domen kutubxonalari: torch/vision/audio/text (2.2.0 ga mos)
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir torchtext==0.17.0

# TTS uchun kerak bo‘lishi mumkin:
RUN pip install --no-cache-dir phonemizer espeakng soundfile librosa pandas tqdm jupyter

# (ixtiyoriy) phonemizer espeak DLL yo‘lini qo‘lda ko‘rsatish uchun:
# ENV ESPEAKNG_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

# PYTHONPATH ni konteyner darajasida to‘g‘rilab qo‘yamiz
ENV PYTHONPATH=/workspace
