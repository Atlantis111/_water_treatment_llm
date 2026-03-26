import gradio as gr
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

# 路径配置
base_model_id = "./models/Qwen3-30B-Instruct"
lora_path = "./models/qwen3-30b-water-lora"
db_dir = "./db/chroma_db"

print("1. 正在初始化 RAG 组件...")
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
retriever = None
if os.path.exists(db_dir):
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("2. 正在加载 30B 基础模型与 LoRA 权重 (BF16)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto", # 单卡会自动映射到 cuda:0
    trust_remote_code=True
)

if os.path.exists(lora_path):
    model = PeftModel.from_pretrained(base_model, lora_path)
    print("✅ 成功加载自定义 LoRA 权重！")
else:
    model = base_model
    print("⚠️ 未找到 LoRA 权重，将使用基础模型进行推理。")
    
model.eval()

def generate_response(message, history):
    messages = [{"role": "system", "content": "你是一个专业的水处理药剂配方推理系统。请根据用户需求和水质报告提供专业的配方，并可以根据用户的反馈随时调整配方。"}]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    context = ""
    if retriever:
        docs = retriever.invoke(message)
        context = "\n".join([d.page_content for d in docs])
    
    if context:
        current_input = f"【参考行业规范】\n{context}\n\n【用户输入】\n{message}"
    else:
        current_input = message

    messages.append({"role": "user", "content": current_input})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

demo = gr.ChatInterface(
    fn=generate_response,
    title="💧 水处理药剂配方系统 (A800 测试版)",
    description="输入水质报告或调整需求，系统将为您推理方案。",
    # theme=gr.themes.Soft() 库更新过后这行主题不能用，去掉用默认的就行
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)