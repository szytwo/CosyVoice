@echo off
chcp 65001 >nul
echo  启动中，请耐心等待 
REM 定义 Anaconda 路径
set CONDA_PATH=D:\ProgramData\anaconda3

REM 激活目标虚拟环境
CALL "%CONDA_PATH%\condabin\conda.bat" activate "D:\AI\CosyVoice\venv"

REM 检查是否激活成功
IF ERRORLEVEL 1 (
    echo 激活虚拟环境失败，请检查路径或环境名称！
    pause
    exit /b
)

REM 设置 GPU 环境变量，选择显卡
set CUDA_VISIBLE_DEVICES=0
set ASR_URL=http://127.0.0.1:7868/api/v1/asr

REM 执行 Python 脚本
python webui.py --model_dir pretrained_models/CosyVoice-300M-Instruct

REM 保持窗口打开
pause
