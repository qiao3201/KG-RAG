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
        
        # 注意：不初始化LLM客户端

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
        print(f"\n【检索过程】")
        print(f"问题: {question}")
        
        # 生成问题嵌入
        query_embedding = self.embedding_model.encode(question)
        
        # 计算相似度
        scores = self.batch_cosine_similarity(query_embedding)
        
        # 获取top-k索引
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        relevant_chunks = []
        print(f"\n找到 {k} 个最相关的知识块:")
        
        for i, idx in enumerate(top_indices, 1):
            chunk_info = {
                'chunk_id': self.chunk_ids[idx],
                'content': self.chunk_contents[idx],
                'source_paper': self.chunk_metadata[idx]['source_paper'],
                'topic': self.chunk_metadata[idx]['topic'],
                'score': float(scores[idx])
            }
            relevant_chunks.append(chunk_info)
            
            print(f"\n  {i}. 知识块ID: {chunk_info['chunk_id']}")
            print(f"     来源文献: 文献{chunk_info['source_paper']}")
            print(f"     主题: {chunk_info['topic']}")
            print(f"     相似度: {chunk_info['score']:.4f}")
            print(f"     内容预览: {chunk_info['content'][:150]}..." if len(chunk_info['content']) > 150 else f"     内容: {chunk_info['content']}")
        
        return relevant_chunks

    def build_context(self, chunks: List[Dict]) -> str:
        """构建上下文字符串"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"文本块{i}：【文献{chunk['source_paper']}】 {chunk['content']}")
        
        return "\n\n".join(context_parts)

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """执行RAG查询（不调用LLM，只返回上下文）"""
        print("=" * 60)
        print("RAG检索系统 - 上下文生成")
        print("=" * 60)
        
        # 1. 检索相关知识块
        relevant_chunks = self.retrieve_relevant_chunks(question, k=k)
        
        if not relevant_chunks:
            print("\n未检索到相关医学文献")
            return {
                'question': question,
                'relevant_chunks': [],
                'context': "无相关医学文献",
                'references': []
            }
        
        # 2. 构建上下文
        context = self.build_context(relevant_chunks)
        
        # 3. 提取参考文献
        source_papers = set()
        for chunk in relevant_chunks:
            source_papers.add(f"文献{chunk['source_paper']}")
        references = "、".join(source_papers)
        
        # 4. 打印最终上下文
        print("\n" + "=" * 60)
        print("【生成的上下文】")
        print("=" * 60)
        print(context)
        print("\n" + "=" * 60)
        print(f"参考文献: {references}")
        print("=" * 60)
        
        return {
            'question': question,
            'relevant_chunks': relevant_chunks,
            'context': context,
            'references': list(source_papers)
        }


def main():
    # 注意：这里不需要设置HF_TOKEN，因为不调用LLM
    # HF_TOKEN = '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'
    # os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 初始化系统
    print("正在初始化RAG系统...")
    rag_system = RAGSystem("config.json", "chunk_vectors.pkl")
    
    # 测试多个问题
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    # 可以选择测试单个问题或所有问题
    #question = "什么是癌因性疲乏？"  # 测试单个问题
    question = questions["4"]  # 测试诊断标准
    
    # 执行查询（不调用LLM）
    result = rag_system.query(question, k=5)
    
    print("\n" + "=" * 60)
    print("检索完成！")
    print(f"问题: {result['question']}")
    print(f"检索到的知识块数量: {len(result['relevant_chunks'])}")
    print(f"参考文献: {'\n'.join(result['references']) if result['references'] else '无'}")


if __name__ == "__main__":
    main()