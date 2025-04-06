# 使用 PyTorch 官方 CUDA 12.1 运行时镜像
# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 设置容器内工作目录为 /workspace
WORKDIR /workspace

# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 防止交互式安装，完全不交互，使用默认值
ENV DEBIAN_FRONTEND=noninteractive
# 设置时区
ENV TZ=Asia/Shanghai

# 更新源并安装依赖
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc g++ make \
    xz-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    unzip sox libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN gcc --version

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 下载并解压 FFmpeg
# https://www.johnvansickle.com/ffmpeg
COPY wheels/linux/ffmpeg-6.0.1-amd64-static.tar.xz .

RUN tar -xJf ffmpeg-6.0.1-amd64-static.tar.xz -C /usr/local \
    && mv /usr/local/ffmpeg-* /usr/local/ffmpeg \
    && rm ffmpeg-6.0.1-amd64-static.tar.xz

# 设置 FFmpeg 到环境变量
ENV PATH="/usr/local/ffmpeg:${PATH}"

# RUN ffmpeg -version

# 设置容器内工作目录为 /code
WORKDIR /code

# 将项目源代码复制到容器中
COPY . /code

# 升级 pip 并安装 Python 依赖：
RUN conda install -y -c conda-forge pynini==2.1.5
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && cd wheels/linux/ \
    && pip install onnxruntime_gpu-1.18.0-cp310-cp310-manylinux_2_28_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && cd /code  \
    && rm -rf /wheels
RUN pip install -r api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN cd pretrained_models/CosyVoice-ttsfrd/ \
    && unzip resource.zip -d . \
    && pip install ttsfrd_dependency-0.1-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && cd /code


# 暴露容器端口
EXPOSE 22
EXPOSE 80
EXPOSE 9987

# 容器启动时执行 api.py
# CMD ["python", "api.py"]
