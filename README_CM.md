## 安装

```
conda create --prefix ./venv python==3.11

conda activate ./venv

conda install -y -c conda-forge pynini==2.1.5

pip install -r ./api_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

docker build -t cosyvoice:20250406 .  # 构建镜像
docker load -i cosyvoice-20250406.tar # 导入镜像
docker save -o cosyvoice-20250406.tar cosyvoice:20250406 # 导出镜像
docker-compose up -d # 后台运行容器

```

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
# third_party/Matcha-TTS
# git clone https://github.com/shivammehta25/Matcha-TTS.git third_party/Matcha-TTS
# 初始化和更新 Git 子模块，确保你在项目的根目录下
git submodule update --init --recursive
```

https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

fastText 语言检测模型
https://fasttext.cc/docs/en/language-identification.html