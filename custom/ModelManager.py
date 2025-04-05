import threading
from custom.CosyVoice import CosyVoice, CosyVoice2
from custom.file_utils import logging

class ModelManager:
    def __init__(self, keep_in_memory=True):
        self.keep_in_memory = keep_in_memory  # 控制模型是否常驻内存
        self.models = {
            "cosyvoice": None,
            "cosyvoice-25hz": None,
            "cosyvoice_sft": None,
            "cosyvoice_instruct": None,
            "cosyvoice2-0.5b": None,
        }
        self.sft_spk = None
        self.locks = {
            "cosyvoice": threading.Lock(),
            "cosyvoice-25hz": threading.Lock(),
            "cosyvoice_sft": threading.Lock(),
            "cosyvoice_instruct": threading.Lock(),
            "cosyvoice2-0.5b": threading.Lock(),
        }

    def _load_model(self, model_type: str):
        """
        内部方法：加载指定类型的模型。
        """
        logging.info(f"Loading model: {model_type}")
        if model_type == "cosyvoice":
            return CosyVoice('pretrained_models/CosyVoice-300M')
        elif model_type == "cosyvoice-25hz":
            return CosyVoice('pretrained_models/CosyVoice-300M-25Hz')
        elif model_type == "cosyvoice_sft":
            model = CosyVoice(
                'pretrained_models/CosyVoice-300M-SFT',
                load_jit=True, load_onnx=False, fp16=True
            )
            self.sft_spk = model.list_avaliable_spks()
            return model
        elif model_type == "cosyvoice_instruct":
            return CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
        elif model_type == "cosyvoice2-0.5b":
            return CosyVoice2(
                'pretrained_models/CosyVoice2-0.5B',
                load_jit=True, load_onnx=False, load_trt=False
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_model(self, model_type: str):
        """
        获取指定类型的模型实例，按需加载，确保线程安全。
        """
        logging.info(f"get_model: {model_type}")
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        if self.keep_in_memory:
            # 常驻模式：双重检查锁定，确保线程安全加载并缓存实例
            if self.models[model_type] is None:
                with self.locks[model_type]:  # 确保线程安全
                    if self.models[model_type] is None:  # 双重检查锁定
                        self.models[model_type] = self._load_model(model_type)

            return self.models[model_type]
        else:
            # 非常驻模式：每次加载新实例，但仍需线程安全
            with self.locks[model_type]:
                return self._load_model(model_type)
