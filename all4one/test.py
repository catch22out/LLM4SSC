from prompt import Prompt
from model import BaseModel
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

class CustomModel(BaseModel):
    def __init__(self, model_name: str, custom_config: dict = None):
        """
        初始化 CustomModel 类。

        参数:
        model_name (str): 预训练模型的名称。
        custom_config (dict): 需要应用于模型的自定义配置。
        """
        super().__init__(model_name)
        self.custom_config = custom_config if custom_config is not None else {}

    def load_model(self):
        """
        加载预训练模型和分词器，并根据自定义配置调整模型设置，然后移动到指定设备。
        """
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 使用自定义配置加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.custom_config,  # 应用自定义配置
            device_map="auto"  # 自动映射设备
        )
        self.model.to(self.device)  # 确保模型在正确的设备上


if __name__ == "__main__":
    # /home/Data/xac/nas/models/models--Phind--Phind-CodeLlama-34B-v2/snapshots/949f61e203f91b412efe8f679c798f09f0ff4b0c
    prompte = Prompt()
    prompte_template = prompte.prompt_generate()

    with open("results.json", "r") as fr:
        results = json.load(fr)
    model_path = "/home/Data/xac/nas/models/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/745795438019e47e4dad1347a0093e11deee4c68"
    model = BaseModel(model_path)
    model.load_model()
    model_predict = model.generate_text(prompt = prompte_template, max_length = 1024)
    results[model_path] = model_predict
    with open("results.json", "w") as fw:
        json.dump(results, fw, indent = 4)
