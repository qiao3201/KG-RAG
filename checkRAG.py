import sys
import io

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import os
import numpy as np
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

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

    def retrieve_relevant_chunks(self, question: str, k: int = 3, verbose: bool = False) -> List[Dict]:
        """检索相关知识块"""
        if verbose:
            print(f"\n【检索过程】")
            print(f"问题: {question}")
            print(f"请求知识块数量: {k}")
        
        # 生成问题嵌入
        query_embedding = self.embedding_model.encode(question)
        
        # 计算相似度
        scores = self.batch_cosine_similarity(query_embedding)
        
        # 获取top-k索引
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        relevant_chunks = []
        
        if verbose:
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
            
            if verbose:
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

    def query(self, question: str, k: int = 3, verbose: bool = False) -> Dict[str, Any]:
        """执行RAG查询（不调用LLM，只返回上下文）"""
        if verbose:
            print("=" * 60)
            print("RAG检索系统 - 上下文生成")
            print("=" * 60)
        
        # 1. 检索相关知识块
        relevant_chunks = self.retrieve_relevant_chunks(question, k=k, verbose=verbose)
        
        if not relevant_chunks:
            if verbose:
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
        
        if verbose:
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


def format_result_for_output(result: Dict[str, Any], question_id: int, k_value: int) -> str:
    """格式化结果为可读的文本"""
    output = []
    output.append(f"{'='*80}")
    output.append(f"问题 ID: {question_id} (k={k_value})")
    output.append(f"问题: {result['question']}")
    output.append(f"{'='*80}")
    
    # 文本块部分
    output.append(f"\n【相关文本块】 (共 {len(result['relevant_chunks'])} 个)")
    for i, chunk in enumerate(result['relevant_chunks'], 1):
        output.append(f"\n  --- 文本块{i} (ID: {chunk['chunk_id']}, 来源: 文献{chunk['source_paper']}, 主题: {chunk['topic']}, 相似度: {chunk['score']:.4f}) ---")
        output.append(f"  {chunk['content']}")
    
    # 参考文献
    output.append(f"\n【参考文献】")
    for ref in result['references']:
        output.append(f"  {ref}")
    
    output.append(f"\n{'='*80}\n\n")
    
    return "\n".join(output)


def main():
    # 初始化系统
    print("正在初始化RAG系统...")
    rag_system = RAGSystem("config.json", "chunk_vectors.pkl")
    
    # 加载问题文件
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 每个问题对应的k值数组
    chunk_amount = [4, 3, 4, 4, 3, 3, 4, 4, 2, 2, 4, 2, 2, 2, 3, 2, 3, 3, 3, 4, 4, 4, 3, 3, 2, 1, 1, 2, 3, 6, 2, 1, 1, 3, 1, 2, 2, 2, 2, 3, 4, 2]
    
    # 打开输出文件
    with open('outcome2.txt', 'w', encoding='utf-8') as out_file:
        out_file.write("RAG系统检索结果 (按问题ID顺序)\n")
        out_file.write(f"{'='*80}\n\n")
        
        # 循环处理问题1-42
        total_questions = 42
        for i in range(1, total_questions + 1):
            question_id = str(i)
            k_value = chunk_amount[i-1]  # 数组索引从0开始
            
            if question_id in questions:
                question = questions[question_id]
                
                print(f"\n正在处理问题 {i}/{total_questions} (k={k_value}): {question[:50]}...")
                
                # 执行查询（不显示详细过程）
                result = rag_system.query(question, k=k_value, verbose=False)
                
                # 格式化结果
                formatted_result = format_result_for_output(result, i, k_value)
                
                # 写入文件
                out_file.write(formatted_result)
                out_file.flush()  # 确保立即写入
                
                print(f"  完成 - 检索到 {len(result['relevant_chunks'])} 个文本块")
            else:
                error_msg = f"警告: 问题 ID {question_id} 不存在\n"
                print(error_msg)
                out_file.write(f"问题 ID {question_id}: 未找到\n\n")
    
    print(f"\n{'='*60}")
    print(f"处理完成！结果已保存到 outcome2.txt")
    print(f"共处理了 {total_questions} 个问题")
    print(f"各问题对应的k值: {chunk_amount}")


if __name__ == "__main__":
    main()