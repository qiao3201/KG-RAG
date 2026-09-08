import json
import os
import sys
import io
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class KGRAGSystem:
    def __init__(self, config_path: str = "config.json"):
        """
        初始化 KG+RAG 系统
        """
        self.load_config(config_path)
        self.init_components()

    def init_components(self):
        """初始化各组件"""
        # 1. 初始化嵌入模型 - 关闭进度条
        print("正在加载嵌入模型...")
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', "BAAI/bge-m3"),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("嵌入模型加载完成")
        
        # 2. 初始化大语言模型客户端
        self.init_llm()
    

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载知识图谱数据
        with open(self.config['kg_data_path'], 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)
        
        # 加载预计算实体向量
        with open(self.config['vectors_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.entity_vectors_raw = data["entity_vectors"]
        
        # 加载知识块数据
        with open(self.config['chunks_path'], 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        # 构建实体查找表
        self.entity_map = {node['id']: node for node in self.kg_data['nodes']}
        
        # 准备实体向量矩阵
        self.prepare_entity_matrix()
    
    def prepare_entity_matrix(self):
        """准备实体向量矩阵用于批量计算"""
        self.entity_ids = []
        self.entity_vectors_matrix = []
        
        # 处理向量数据
        for entity_id, vector in self.entity_vectors_raw.items():
            if isinstance(vector, list):
                self.entity_ids.append(entity_id)
                self.entity_vectors_matrix.append(np.array(vector, dtype=np.float32))
            else:
                print(f"警告: 实体 {entity_id} 的值类型为 {type(vector)}，跳过")
        
        if self.entity_vectors_matrix:
            self.entity_vectors_matrix = np.array(self.entity_vectors_matrix, dtype=np.float32)
            print(f"实体向量矩阵准备完成: {len(self.entity_ids)} 个实体")
            print(f"向量矩阵形状: {self.entity_vectors_matrix.shape}")
        else:
            print("错误: 没有有效的向量数据")
    
    def init_llm(self):
        """初始化 Hugging Face Inference API 客户端"""
        if "HF_TOKEN" not in os.environ:
            raise ValueError("请设置环境变量 HF_TOKEN")
        
        self.llm_client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        self.llm_model = self.config.get('llm_model', "Qwen/Qwen3-4B-Instruct-2507:featherless-ai")
        print(f"已连接到模型: {self.llm_model}")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def batch_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # 归一化
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # 计算所有向量的范数
        vector_norms = np.linalg.norm(self.entity_vectors_matrix, axis=1, keepdims=True)
        vector_norms = np.where(vector_norms == 0, 1e-8, vector_norms)
        vectors_norm = self.entity_vectors_matrix / vector_norms
        
        # 批量计算
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities
    
    def get_relevant_entities(self, question: str, k: int = 10) -> List[Dict]:
        """获取相关问题实体"""
        # 生成问题嵌入
        query_embedding = self.embedding_model.encode(question)
        
        # 批量计算相似度
        scores = self.batch_cosine_similarity(query_embedding)
        
        # 获取top-k索引
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        relevant_entities = []
        for idx in top_indices:
            entity_id = self.entity_ids[idx]
            node = self.entity_map[entity_id]
            
            degree = node.get('degree', 0)
            if isinstance(degree, str):
                degree = int(degree) if degree.isdigit() else 0
            
            relevant_entities.append({
                'id': entity_id,
                'name': node['name'],
                'type': node['type'],
                'topic': node['topic'],
                'degree': degree,
                'score': float(scores[idx])
            })
        
        return relevant_entities
    
    def get_related_triplets(self, entity_ids: List[str]) -> List[Dict]:
        """获取相关三元组 - 纳入直接连接和2跳连接"""
        related_triplets = []
        added_edges = set()
        entity_ids_set = set(entity_ids)
        
        # 构建邻接表和边映射
        adjacency = {}
        edge_map = {}
        
        for edge in self.kg_data['edges']:
            # 构建双向邻接表
            if edge['source'] not in adjacency:
                adjacency[edge['source']] = []
            adjacency[edge['source']].append(edge['target'])
            
            if edge['target'] not in adjacency:
                adjacency[edge['target']] = []
            adjacency[edge['target']].append(edge['source'])
            
            # 存储边信息
            edge_key = f"{edge['source']}-{edge['target']}"
            edge_map[edge_key] = edge
        
        # 1. 直接连接
        for edge in self.kg_data['edges']:
            if edge['source'] in entity_ids_set and edge['target'] in entity_ids_set:
                edge_key = f"{edge['source']}-{edge['target']}"
                if edge_key not in added_edges:
                    source_node = self.entity_map.get(edge['source'])
                    target_node = self.entity_map.get(edge['target'])
                    if source_node and target_node:
                        related_triplets.append({
                            'source': source_node['name'],
                            'target': target_node['name'],
                            'relation': edge['relation'],
                            'source_chunk': edge.get('source_chunk', '')
                        })
                        added_edges.add(edge_key)
        
        # 2. 2跳连接
        entity_list = list(entity_ids_set)
        for i, a in enumerate(entity_list):
            for b in entity_list[i+1:]:
                # 查找共同邻居
                neighbors_a = set(adjacency.get(a, []))
                neighbors_b = set(adjacency.get(b, []))
                common_neighbors = neighbors_a & neighbors_b
                
                for mid in common_neighbors:
                    # 添加边 a-mid
                    edge_key1 = f"{a}-{mid}"
                    if edge_key1 in edge_map and edge_key1 not in added_edges:
                        edge = edge_map[edge_key1]
                        source_node = self.entity_map.get(edge['source'])
                        target_node = self.entity_map.get(edge['target'])
                        if source_node and target_node:
                            related_triplets.append({
                                'source': source_node['name'],
                                'target': target_node['name'],
                                'relation': edge['relation'],
                                'source_chunk': edge.get('source_chunk', '')
                            })
                            added_edges.add(edge_key1)
                    
                    # 添加边 mid-b
                    edge_key2 = f"{mid}-{b}"
                    if edge_key2 in edge_map and edge_key2 not in added_edges:
                        edge = edge_map[edge_key2]
                        source_node = self.entity_map.get(edge['source'])
                        target_node = self.entity_map.get(edge['target'])
                        if source_node and target_node:
                            related_triplets.append({
                                'source': source_node['name'],
                                'target': target_node['name'],
                                'relation': edge['relation'],
                                'source_chunk': edge.get('source_chunk', '')
                            })
                            added_edges.add(edge_key2)
        
        return related_triplets
    
    def get_chunks_by_triplets(self, triplets: List[Dict]) -> List[Dict]:
        """根据三元组获取相关知识块"""
        chunk_ids = set()
        for triplet in triplets:
            if triplet.get('source_chunk'):
                chunk_ids.add(str(triplet['source_chunk']))
        
        chunks = []
        
        for chunk in self.chunks_data:
            chunk_id_str = str(chunk['chunk_id'])
            
            if chunk_id_str in chunk_ids:
                chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'],
                    'source_paper': chunk['source_paper']
                })
        
        return chunks
    
    def build_context(self, question: str) -> str:
        """构建上下文 - 返回检索到的文本块列表"""
        # 1. 获取相关实体
        relevant_entities = self.get_relevant_entities(question, k=self.config.get('k_entities', 10))
        
        if not relevant_entities:
            return "无相关医学文献"
        
        # 2. 获取相关三元组
        entity_ids = [e['id'] for e in relevant_entities]
        triplets = self.get_related_triplets(entity_ids)
        
        # 3. 获取相关知识块
        chunks = self.get_chunks_by_triplets(triplets)
        
        if not chunks:
            return "无相关医学文献"
        
        # 4. 构建文本块列表
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"文本块{i}：【文献{chunk['source_paper']}】 {chunk['content']}")
        
        return "\n\n".join(context_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        # 构建上下文
        context = self.build_context(question)
        
        # 获取相关实体
        relevant_entities = self.get_relevant_entities(question, k=self.config.get('k_entities', 10))
        
        triplets = []
        chunks = []
        source_papers = set()
        
        if relevant_entities:
            entity_ids = [e['id'] for e in relevant_entities]
            triplets = self.get_related_triplets(entity_ids)
            chunks = self.get_chunks_by_triplets(triplets)
            
            # 收集来源文献
            for chunk in chunks:
                if chunk.get('source_paper'):
                    source_papers.add(f"文献{chunk['source_paper']}")
        
        # 格式化参考文献字符串
        references = "、".join(source_papers) if source_papers else "医学文献"
        
        # 构建提示词
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
            'context': context,
            'triplets': triplets,
            'chunks': chunks,
            'references': list(source_papers),
            'entities': relevant_entities
        }