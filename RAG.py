import sys
import io

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import os
import sys
import io
import numpy as np
import pickle
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class RAGSystem:
    def __init__(self, config_path: str = "config.json", vector_file: str = "chunk_vectors.pkl"):
        """
        初始化 RAG 系统（基于知识块检索）
        
        Args:
            config_path: 配置文件路径
            vector_file: 预计算向量文件路径
        """
        self.vector_file = vector_file
        self.load_config(config_path)
        self.init_components()
        self.load_chunk_vectors()

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载知识块数据（用于获取原始内容，如果需要）
        with open(self.config['chunks_path'], 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        print(f"知识块数据加载完成: {len(self.chunks_data)} 个知识块")

    def init_components(self):
        """初始化各组件"""
        # 1. 初始化嵌入模型（用于查询向量化）
        print("正在加载嵌入模型...")
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', "BAAI/bge-m3"),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("嵌入模型加载完成")
        
        # 2. 初始化大语言模型客户端
        self.init_llm()

    def init_llm(self):
        """初始化 Hugging Face Inference API 客户端"""
        if "HF_TOKEN" not in os.environ:
            raise ValueError("请设置环境变量 HF_TOKEN")
        
        self.llm_client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        self.llm_model = self.config.get('llm_model', "Qwen/Qwen3-4B-Instruct-2507:nscale")
        print(f"已连接到模型: {self.llm_model}")

    def load_chunk_vectors(self):
        """加载预计算的知识块向量"""
        print(f"正在加载预计算向量: {self.vector_file}")
        
        with open(self.vector_file, 'rb') as f:
            vector_data = pickle.load(f)
        
        self.chunk_ids = vector_data['ids']
        self.chunk_contents = vector_data['contents']
        self.chunk_metadata = vector_data['metadata']
        self.chunk_vectors = np.array(vector_data['vectors'], dtype=np.float32)
        
        print(f"向量数据加载完成:")
        print(f"  - 知识块数量: {len(self.chunk_ids)}")
        print(f"  - 向量矩阵形状: {self.chunk_vectors.shape}")

    def batch_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # 归一化
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        vectors_norm = self.chunk_vectors / (np.linalg.norm(self.chunk_vectors, axis=1, keepdims=True) + 1e-8)
        
        # 批量计算
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities

    def retrieve_relevant_chunks(self, question: str, k: int = 3) -> List[Dict]:
        """检索相关知识块"""
        # 生成问题嵌入
        query_embedding = self.embedding_model.encode(question)
        
        # 计算相似度
        scores = self.batch_cosine_similarity(query_embedding)
        
        # 获取top-k索引
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'chunk_id': self.chunk_ids[idx],
                'content': self.chunk_contents[idx],
                'source_paper': self.chunk_metadata[idx]['source_paper'],
                'topic': self.chunk_metadata[idx]['topic'],
                'score': float(scores[idx])
            })
        
        return relevant_chunks

    def build_context(self, chunks: List[Dict]) -> str:
        """构建上下文字符串"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"文本块{i}：【文献{chunk['source_paper']}】 {chunk['content']}")
        
        return "\n\n".join(context_parts)

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """执行RAG查询"""
        # 1. 检索相关知识块
        relevant_chunks = self.retrieve_relevant_chunks(question, k=k)
        
        if not relevant_chunks:
            return {
                'question': question,
                'answer': "抱歉，未检索到相关医学文献。",
                'relevant_chunks': [],
                'references': []
            }
        
        # 2. 构建上下文
        context = self.build_context(relevant_chunks)
        
        # 3. 提取参考文献
        source_papers = set()
        for chunk in relevant_chunks:
            source_papers.add(f"文献{chunk['source_paper']}")
        references = "、".join(source_papers)
        
        # 4. 构建提示词
        prompt = f"""你是一位专业的肿瘤护理专家，专门负责肺癌患者的癌因性疲乏（CRF）管理。请严格依据以下提供的医学文献片段，回答患者问题。

【注意】
1.请勿引入外部知识。
2.请确保回答专业、准确、清晰。
3.简要回答，字数在500内。

【检索到的文本块列表】
{context}

【患者问题】
{question}

请基于上述文献内容，以结构化的方式回答，包括：
- 核心答案
- 注意事项
- 参考文献：{references}"""
        
        try:
            messages = [
                {"role": "system", "content": "你是一位专业的肿瘤护理专家，专门负责肺癌患者的癌因性疲乏（CRF）管理。"},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config.get('max_tokens', 500),
                temperature=self.config.get('temperature', 0.1)
            )
            
            response = completion.choices[0].message.content
            
        except Exception as e:
            print(f"调用API时出错: {e}")
            response = f"抱歉，生成回答时出现错误: {str(e)}"
        
        return {
            'question': question,
            'answer': response,
            'context': context,  # 添加上下文
            'relevant_chunks': relevant_chunks,
            'references': list(source_papers)
        }


def main():
    # 设置HF_TOKEN
    HF_TOKEN = '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'
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 初始化系统
    rag_system = RAGSystem("config.json", "chunk_vectors.pkl")
    
    # 加载问题文件
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    total_questions = 42
    repeat_times = 14
    
    # 重复执行14次
    for run in range(3, repeat_times + 1):
        output_file = f"RAG{run}.txt"
        print(f"\n{'='*60}")
        print(f"第 {run} 次运行，结果将保存到 {output_file}")
        print(f"{'='*60}")
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(f"RAG系统检索结果 - 第{run}次运行\n")
            out_file.write("包含：问题 + 文本块 + LLM回答\n")
            out_file.write(f"{'='*80}\n\n")
            
            # 循环处理问题1-42
            for i in range(1, total_questions + 1):
                question_id = str(i)
                if question_id in questions:
                    question = questions[question_id]
                    
                    print(f"  正在处理问题 {i}/{total_questions}...")
                    
                    # 执行查询
                    result = rag_system.query(question, k=3)
                    
                    # 输出格式：问题 + 文本块 + LLM回答
                    out_file.write(f"【问题 {i}】\n")
                    out_file.write(f"{result['question']}\n\n")
                    
                    out_file.write("【检索到的文本块】\n")
                    if result['relevant_chunks']:
                        for j, chunk in enumerate(result['relevant_chunks'], 1):
                            out_file.write(f"  文本块{j}：【文献{chunk['source_paper']}】 {chunk['content']}\n\n")
                    else:
                        out_file.write("  未检索到相关文本块\n\n")
                    
                    out_file.write("【LLM回答】\n")
                    out_file.write(f"{result['answer']}\n\n")
                    
                    # 添加分隔线
                    out_file.write(f"{'-'*60}\n\n")
                    out_file.flush()  # 确保立即写入
                    
                else:
                    out_file.write(f"【问题 {i}】\n")
                    out_file.write("未找到对应问题\n\n")
                    out_file.write(f"{'-'*60}\n\n")
            
            print(f"  第 {run} 次运行完成，已保存到 {output_file}")
    
    print(f"\n{'='*60}")
    print(f"全部完成！共执行 {repeat_times} 次")
    print(f"生成文件: RAG1.txt 到 RAG{repeat_times}.txt")
    print(f"每次处理 {total_questions} 个问题")

if __name__ == "__main__":
    main()