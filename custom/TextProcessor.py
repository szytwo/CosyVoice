import datetime
import json
import os
import re
import traceback
from datetime import datetime

import fasttext

from custom.file_utils import logging


class TextProcessor:
    """
    文本处理工具类，提供多种文本相关功能。
    """

    @staticmethod
    def clear_text(text):
        text = text.replace("\n", "")
        text = TextProcessor.replace_corner_mark(text)
        return text

    # replace special symbol
    @staticmethod
    def replace_corner_mark(text):
        text = text.replace('²', '平方')
        text = text.replace('³', '立方')
        return text

    @staticmethod
    def detect_language(text):
        """
        检测输入文本的语言。
        :param text: 输入文本
        :return: 返回检测到的语言代码（如 'en', 'zh', 'ja', 'ko'）
        """

        # 加载预训练的语言检测模型
        fasttext_model = fasttext.load_model("./fasttext/lid.176.bin")

        try:
            lang = None
            text = text.strip()
            if text:
                predictions = fasttext_model.predict(text, k=1)  # 获取 top-1 语言预测
                lang = predictions[0][0].replace("__label__", "")  # 解析语言代码
                confidence = predictions[1][0]  # 置信度
                lang = lang if confidence > 0.6 else None

            logging.info(f'Detected language: {lang}')
            return lang
        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return None

    @staticmethod
    def ensure_sentence_ends_with_period(text, add_lang_tag: bool = False):
        """
        确保输入文本以适当的句号结尾。
        :param text: 输入文本
        :param add_lang_tag: 是否添加语言标签
        :return: 修改后的文本
        """
        if not text.strip():
            return text, None  # 空文本直接返回
        # 根据文本内容添加适当的句号
        lang = TextProcessor.detect_language(text)
        lang_tag = ''
        if add_lang_tag:
            if lang == 'zh' or lang == 'zh-cn':  # 中文文本
                lang_tag = '<|zh|>'
            elif lang == 'en':  # 英语
                lang_tag = '<|en|>'
            elif lang == 'ja':  # 日语
                lang_tag = '<|jp|>'
            elif lang == 'ko':  # 韩语
                lang_tag = '<|ko|>'
        # 判断是否已经以句号结尾
        if text[-1] in ['.', '。', '！', '!', '？', '?']:
            return f'{lang_tag}{text}', lang
        # 根据文本内容添加适当的句号
        if lang == 'zh' or lang == 'zh-cn' or lang == 'ja':  # 中文文本
            return f'{lang_tag}{text}。', lang
        else:  # 英文或其他
            return f'{lang_tag}{text}.', lang

    @staticmethod
    def log_error(exception: Exception, log_dir='error'):
        """
        记录错误信息到指定目录，并按日期时间命名文件。

        :param exception: 捕获的异常对象
        :param log_dir: 错误日志存储的目录，默认为 'error'
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 获取当前时间戳，格式化为 YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 创建日志文件路径
        log_file_path = os.path.join(log_dir, f'error_{timestamp}.log')
        # 使用 traceback 模块获取详细的错误信息
        error_traceback = traceback.format_exc()
        # 写入错误信息到文件
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"错误发生时间: {timestamp}\n")
            log_file.write(f"错误信息: {str(exception)}\n")
            log_file.write("堆栈信息:\n")
            log_file.write(error_traceback + '\n')

        logging.error(f"错误信息: {str(exception)}\n"
                      f"详细信息已保存至: {log_file_path}")

    @staticmethod
    def get_keywords(config_file='.\custom\keywords.json'):
        with open(config_file, 'r', encoding='utf-8') as file:
            words_list = json.load(file)
        return words_list

    # noinspection PyTypeChecker
    @staticmethod
    def add_quotation_mark(text, keywords, min_length=2):
        """
        在文本中为指定的词语添加引号，跳过长度小于 min_length 的词语。

        :param text: 输入文本
        :param keywords: 需要添加括号的词语列表
        :param min_length: 跳过添加括号的最小词语长度，默认为 2
        :return: 处理后的文本
        """

        text = text.replace("\n", "")
        text = TextProcessor.replace_blank(text)
        text = TextProcessor.replace_bracket(text)
        text = TextProcessor.replace_corner_mark(text)
        logging.info(f'add quotation original text: {text}')

        # 常见引号标点符号
        punctuation = r'[\[\]（）【】《》““””‘’]'
        # 分割文本为引号内外的部分
        split_pattern = r'(“.*?”)'  # 非贪婪匹配引号内的内容
        # 按关键词长度从长到短排序
        keywords = sorted(keywords, key=len, reverse=True)

        for word in keywords:
            if len(word) >= min_length:
                parts = re.split(split_pattern, text)
                for i in range(len(parts)):
                    # 处理引号外的部分（偶数索引）
                    if i % 2 == 0:
                        current_part = parts[i]
                        # 匹配时确保目标词前后没有标点符号，且没有已有的引号
                        pattern = rf'(?<!“)(?<!{punctuation}){re.escape(word)}(?!{punctuation})(?<!”)'
                        # 使用正则表达式替换
                        current_part = re.sub(pattern, f'“{word}”', current_part, flags=re.IGNORECASE)
                        parts[i] = current_part
                # 合并所有部分
                text = ''.join(parts)

        logging.info(f'add quotation out text: {text}')

        return text

    # replace meaningless symbol
    @staticmethod
    def replace_bracket(text):
        text = text.replace('（', '“').replace('）', '”')
        text = text.replace('【', '“').replace('】', '”')
        return text

    # remove blank between chinese character
    # noinspection PyTypeChecker
    @staticmethod
    def replace_blank(text: str):
        out_str = []
        for i, c in enumerate(text):
            if c == " ":
                if ((text[i + 1].isascii() and text[i + 1] != " ") and
                        (text[i - 1].isascii() and text[i - 1] != " ")):
                    out_str.append(c)
            else:
                out_str.append(c)
        return "".join(out_str)

    @staticmethod
    def number_to_chinese(num_str):
        """将数字字符串转换为中文读法"""
        chinese_digits = {'0': '零', '1': '一', '2': '二', '3': '三',
                          '4': '四', '5': '五', '6': '六', '7': '七',
                          '8': '八', '9': '九'}
        return ''.join(chinese_digits[c] for c in num_str)

    # noinspection PyTypeChecker
    @staticmethod
    def replace_chinese_year(text, keywords):
        """将数字年转换为中文读法"""
        logging.info(f'replace chinese year original text: {text}')

        def generate_year_range():
            """生成当前年及前后一年（共3个年份）"""
            current = datetime.now().year
            return {
                str(current - 1),
                str(current),
                str(current + 1)
            }

        # 生成核心年份集合（用户提供+当前年及前后）
        keywords = set(keywords) | generate_year_range()  # 自动合并去重
        # 生成年份映射表（仅处理纯数字关键词）
        year_map = {
            year: TextProcessor.number_to_chinese(year)
            for year in set(keywords)
            if year.isdigit()  # 过滤确保是数字
        }

        if year_map:
            # 按长度倒序排列，避免短数字优先匹配（如同时有202和2025）
            sorted_years = sorted(year_map.keys(), key=lambda x: -len(x))
            # 仅匹配自然语言边界（排除数学符号）
            year_pattern = r'(?<!\d)({})(年|(?=[,，。！？”’\"\'\s]|$))'.format(
                "|".join(map(re.escape, sorted_years))
            )

            # 单次替换同时处理带"年"和单独出现的情况
            def replace_year(match):
                num, suffix = match.groups()
                return year_map[num] + ("年" if suffix == "年" else "")

            text = re.sub(
                year_pattern,
                replace_year,
                text
            )

        logging.info(f'replace chinese year out text: {text}')

        return text
