import os
import requests
from custom.file_utils import logging, get_full_path
from custom.TextProcessor import TextProcessor

class AsrProcessor:
    def __init__(self):
        """
        初始化ASR音频与文本对齐处理器。
        """
        asr_url = os.getenv("ASR_URL", "") #asr接口
        self.asr_url = asr_url

    def send_asr_request(self, audio_path, lang='auto', output_timestamp=False):
        # 发送 GET 请求
        params = {'audio_path': audio_path, 'lang': lang, 'output_timestamp': output_timestamp}
        headers = {'accept': 'application/json'}

        response = requests.get(self.asr_url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json()  # 返回 JSON 响应
        else:
            logging.error(f"send_asr_request fail: {response.status_code}")
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
