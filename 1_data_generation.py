import os
import json
from vllm import LLM, SamplingParams
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 初始化 vLLM (Qwen3-72B) - 假设已量化或有多卡支持
llm = LLM(
    model="Qwen/Qwen3-72B-Instruct-AWQ", # 依然需要量化版本
    tensor_parallel_size=1, 
    trust_remote_code=True,
    quantization="AWQ",
    max_model_len=4096 
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

# 2. 简易 RAG 设置 (读取水处理专业知识)
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
if os.path.exists("./data/rules"):
    docs = []
    for file in os.listdir("./data/rules"):
        loader = TextLoader(os.path.join("./data/rules", file), encoding='utf-8')
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
else:
    retriever = None

def generate_dataset():
    report_dir = "./data/raw_reports"
    output_file = "./data/dataset.jsonl"
    
    reports = [f for f in os.listdir(report_dir) if f.endswith('.txt')]
    dataset = []

    for report_name in reports:
        with open(os.path.join(report_dir, report_name), 'r', encoding='utf-8') as f:
            report_content = f.read()
            
        # RAG 检索背景知识
        context = ""
        if retriever:
            docs = retriever.invoke(report_content)
            context = "\n".join([d.page_content for d in docs])

        # 构建 Prompt
        system_prompt = "你是一个顶级的水处理工程师。请根据用户提供的水质报告，结合以下知识库，推理出最合适的水处理药剂配方。"
        user_prompt = f"【参考知识】\n{context}\n\n【水质报告】\n{report_content}\n\n请给出药剂配方，并说明理由。"

        # 调用 72B 生成
        prompts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"]
        outputs = llm.generate(prompts, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # 构造微调所需的 ChatML 格式数据
        data_entry = {
            "messages": [
                {"role": "system", "content": "你是一个水处理专家。请根据水质报告提供药剂配方。"},
                {"role": "user", "content": report_content},
                {"role": "assistant", "content": generated_text}
            ]
        }
        dataset.append(data_entry)

    # 保存 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"数据生成完毕，共 {len(dataset)} 条，保存在 {output_file}")

if __name__ == "__main__":
    generate_dataset()