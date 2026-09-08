import sys
import io

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch
import os

class ChunkVectorBuilder:
    def __init__(self, config_path: str = "config.json"):
        """初始化知识块向量构建器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载知识块数据
        with open(self.config['chunks_path'], 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        print(f"知识块数据加载完成: {len(self.chunks_data)} 个知识块")
        
        # 初始化嵌入模型
        print("正在加载嵌入模型...")
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', "BAAI/bge-m3"),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("嵌入模型加载完成")
    
    def compute_vectors(self):
        """计算所有知识块的向量"""
        print("正在为知识块计算向量...")
        
        chunk_ids = []
        chunk_contents = []
        chunk_metadata = []
        chunk_vectors = []
        
        for i, chunk in enumerate(self.chunks_data):
            # 存储元数据
            chunk_ids.append(chunk['chunk_id'])
            chunk_contents.append(chunk['content'])
            chunk_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'source_paper': chunk['source_paper'],
                'topic': chunk.get('topic', '')
            })
            
            # 计算向量
            vector = self.embedding_model.encode(chunk['content'])
            chunk_vectors.append(vector)
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(self.chunks_data)} 个知识块")
        
        # 转换为numpy数组
        chunk_vectors = np.array(chunk_vectors, dtype=np.float32)
        
        print(f"知识块向量计算完成: {len(chunk_ids)} 个知识块")
        print(f"向量矩阵形状: {chunk_vectors.shape}")
        
        return {
            'ids': chunk_ids,
            'contents': chunk_contents,
            'metadata': chunk_metadata,
            'vectors': chunk_vectors
        }
    
    def save_vectors(self, vector_data: dict, filename: str = "chunk_vectors.pkl"):
        """保存向量数据到文件"""
        # 准备保存的数据（将numpy数组转为列表以便pickle序列化）
        save_data = {
            'ids': vector_data['ids'],
            'contents': vector_data['contents'],
            'metadata': vector_data['metadata'],
            'vectors': vector_data['vectors'].tolist()  # numpy数组转列表
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"向量数据已保存到: {filename}")
        print(f"  - 知识块数量: {len(vector_data['ids'])}")
        print(f"  - 向量维度: {vector_data['vectors'].shape[1]}")
    
    def build_and_save(self):
        """构建并保存向量"""
        vector_data = self.compute_vectors()
        self.save_vectors(vector_data, "chunk_vectors.pkl")
        print("知识块向量构建完成！")


def main():
    builder = ChunkVectorBuilder("config.json")
    builder.build_and_save()


if __name__ == "__main__":
    main()