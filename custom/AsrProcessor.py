import os
import requests

from custom.TextProcessor import TextProcessor
from custom.file_utils import logging, get_full_path


class AsrProcessor:
    def __init__(self):
        """
        初始化ASR音频与文本对齐处理器。
        """
        asr_url = os.getenv("ASR_URL", "")  # asr接口
        self.asr_url = asr_url

    def send_asr_request(self, audio_path, lang='auto', output_timestamp=False):
        """
        通过 POST 上传音频文件到 ASR 服务

        Args:
            audio_path (str): 本地音频文件路径（如 /path/to/audio.wav）
            lang (str): 语言代码（默认 'auto' 自动检测）
            output_timestamp (bool): 是否返回时间戳

        Returns:
            dict: ASR 结果（JSON 格式），失败返回 None
        """
        try:
            with open(audio_path, 'rb') as audio_file:
                files = [('files', (os.path.basename(audio_path), audio_file, 'audio/wav'))]
                data = {
                    'keys': os.path.basename(audio_path),
                    'lang': lang,
                    'output_timestamp': str(output_timestamp).lower()
                }

                response = requests.post(
                    self.asr_url,
                    files=files,
                    data=data,
                    headers={'accept': 'application/json'}
                )

            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"ASR failed. Status: {response.status_code}, Response: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error in send_asr_request: {str(e)}")
            return None

    def asr_to_text(self, audio_path):
        try:
            logging.info(f"正在使用 ASR 进行音频转文本...")
            # 构建保存路径
            audio_path = get_full_path(audio_path)
            # 发送 ASR 请求并获取识别结果
            result = self.send_asr_request(audio_path)

            if result:
                logging.info("ASR 音频转文本完成!")
                return result['result'][0]['clean_text']
        except Exception as e:
            TextProcessor.log_error(e)

        logging.error("ASR 音频转文本失败!")
        return None
