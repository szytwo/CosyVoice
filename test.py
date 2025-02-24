from custom.TextProcessor import TextProcessor

# 测试
print(TextProcessor.detect_language("你好，我是中国人"))  # 预计输出: zh
print(TextProcessor.detect_language("Hello, how are you?"))  # 预计输出: en
print(TextProcessor.detect_language("こんにちは、お元気ですか？"))  # 预计输出: ja
print(TextProcessor.detect_language("안녕하세요"))  # 预计输出: ko
