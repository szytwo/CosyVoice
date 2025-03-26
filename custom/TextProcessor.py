import datetime
import json
import os
import re
import traceback
from datetime import datetime

import cn2an
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

    # noinspection PyTypeChecker
    @staticmethod
    def replace_chinese_number(text):
        """
        将文本中的数字和单位替换为中文读法。
        :param text: 输入文本。
        :return: 替换后的文本。
        """
        logging.info(f'replace chinese number original text: {text}')

        def smart_convert(input_str):
            """
            根据输入字符串智能转换数字部分。
            :param input_str: 输入字符串（如 "2003计划"、"20年"、"2008份"）。
            :return: 转换后的中文读法。
            """
            if not input_str:
                return input_str
            # 检查是否有后缀
            for suffix, rule in suffix_rules.items():
                if input_str.endswith(suffix):
                    num_part = input_str[:-len(suffix)]  # 去掉后缀
                    if "lengths" in rule and len(num_part) not in rule["lengths"]:
                        # 如果长度不符合规则，按普通数字转换
                        chinese_num = cn2an.an2cn(num_part)
                    else:
                        # 按规则中的模式转换
                        chinese_num = cn2an.an2cn(num_part, mode=rule["mode"])
                    return chinese_num + suffix  # 拼接后缀
            # 如果没有后缀且是4位数字，按逐字符转换
            if input_str.isdigit() and len(input_str) == 4:
                return cn2an.an2cn(input_str, mode="direct")
            # 其他情况按普通数字转换
            return cn2an.an2cn(input_str)

        # 排除符号
        exclude_symbols = "%+-/*=$|℃"
        # 逐字符转换的单位
        direct_units = ["年"]
        # 普通数字转换的单位
        low_units = ["月", "日", "小时", "分钟", "秒", "个", "人", "次", "份", "元", "美元", "米", "千克", "升", "遍",
                     "件", "瓶", "款", "道"]
        # 动态生成 suffix_rules
        suffix_rules = {}
        for unit in direct_units:
            suffix_rules[unit] = {"lengths": [4, 4], "mode": "direct"}  # 逐字符转换
        for unit in low_units:
            suffix_rules[unit] = {"mode": "low"}  # 普通数字转换
        # 构建单位正则表达式
        units_pattern = "|".join(direct_units + low_units)  # 正则表达式匹配数字部分（包括带单位和不带单位的情况）
        # 正则表达式匹配数字部分（包括带单位和不带单位的情况）
        pattern = re.compile(
            rf"\d+(?:{units_pattern})|(?<!\d)\d{{4}}(?![{units_pattern}{re.escape(exclude_symbols)}\d])"
        )

        def repl_text(m):
            s = m.group(0)
            # 如果包含排除符号，则不替换
            if any(symbol in s for symbol in exclude_symbols):
                return s

            try:
                return smart_convert(s)
            except Exception as e:
                TextProcessor.log_error(e)
                logging.error(f"replace chinese number repl text error：{s}\n{str(e)}")
                return s

        text = pattern.sub(repl_text, text)

        logging.info(f'replace chinese number out text: {text}')

        return text

    @staticmethod
    def replace_pronunciation(text, keywords):
        """替换文本中的发音错误字"""
        for wrong_char, correct_char in keywords.items():
            text = text.replace(wrong_char, correct_char)

        return text
