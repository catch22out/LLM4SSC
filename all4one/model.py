import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from prompt import Prompt
import json
# 配置环境变量
# os.environ["HF_DATASETS_CACHE"] = "/home/Data/xac/nas/models"
# os.environ["HF_HOME"] = "/home/Data/xac/nas/models"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/Data/xac/nas/models"
# os.environ["TRANSFORMERS_CACHE"] = "/home/Data/xac/nas/models"

class BaseModel:
    def __init__(self, model_name: str):
        """
        初始化 BaseModel 类。

        参数:
        model_name (str): 预训练模型的名称。
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """
        加载预训练模型和分词器，并移动到指定设备。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def save_model(self, save_path: str):
        """
        保存模型和分词器到指定路径。

        参数:
        save_path (str): 模型和分词器的保存路径。
        """
        self.model.module.save_pretrained(save_path) if isinstance(self.model, torch.nn.DataParallel) else self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}.")

    def generate_text(self, prompt: str, max_length: int = 500):
        """
        根据提示生成文本。

        参数:
        prompt (str): 文本生成的提示语。
        max_length (int): 生成文本的最大长度。

        返回:
        str: 生成的文本。
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def fine_tune(self, dataset_name: str, output_dir: str, num_train_epochs: int = 1):
        """
        微调模型。

        参数:
        dataset_name (str): 数据集的名称。
        output_dir (str): 微调后模型的保存路径。
        num_train_epochs (int): 训练的轮数。
        """
        dataset = load_dataset(dataset_name)
        
        def preprocess_function(examples):
            """
            预处理函数，将文本进行分词。

            参数:
            examples (dict): 包含文本的字典。

            返回:
            dict: 分词后的字典。
            """
            return self.tokenizer(examples['text'], truncation=True)

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation']
        )

        trainer.train()
        self.save_model(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}.")