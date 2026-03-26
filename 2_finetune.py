import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 建议将这里改为你本地的模型路径，如 "/data/models/Qwen3-30B-Instruct"
model_id = "./models/Qwen3-30B-Instruct" 
output_dir = "./models/qwen3-30b-water-lora"

def train():
    print("1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("2. 加载数据集...")
    dataset = load_dataset("json", data_files="./data/dataset.jsonl", split="train")

    print("3. 配置 4-bit 量化 (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map={"": 0}, # 绑定到单卡 A800
        trust_remote_code=True
    )
    
    # 开启梯度检查点节省显存
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=3, 
        save_strategy="epoch",
        bf16=True, 
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=2048  # <--- 名字缩短了！
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, # ✅ 修改点 2：为防止下一步报错，把 tokenizer 改为 processing_class
        args=training_args,
    )

    print("4. 开始在 A800 上进行微调训练...")
    trainer.train()
    
    print("5. 保存 LoRA 权重...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ 训练完成！")

if __name__ == "__main__":
    train()