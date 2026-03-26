import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter 这个库已经版本大重构，没有了，换成下面的
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_kb():
    knowledge_dir = "./data/rules"
    db_dir = "./db/chroma_db"
    
    if not os.path.exists(knowledge_dir) or not os.listdir(knowledge_dir):
        print("⚠️ 知识库目录为空，跳过向量库构建。")
        return

    documents = []
    for file in os.listdir(knowledge_dir):
        file_path = os.path.join(knowledge_dir, file)
        if file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # 首次运行会自动下载这个轻量级的中文 Embedding 模型
    embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    
    Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=db_dir)
    print(f"✅ 知识库构建完毕，已保存至 {db_dir}")

if __name__ == "__main__":
    build_kb()