import json
import os
import sys
import io
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
        
        # 2. 注意：不再初始化LLM客户端

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
            # 关键：取出 "entity_vectors" 这个键的值
            self.entity_vectors_raw = data["entity_vectors"]  # 或者 data.get("entity_vectors", {})
        
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
            # 直接使用vector，如果是列表就正常处理
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
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def batch_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        # 确保query_vector是numpy数组且类型正确
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # 归一化
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)  # 加一个小数防止除0
        
        # 计算所有向量的范数
        vector_norms = np.linalg.norm(self.entity_vectors_matrix, axis=1, keepdims=True)
        vector_norms = np.where(vector_norms == 0, 1e-8, vector_norms)  # 处理零向量
        vectors_norm = self.entity_vectors_matrix / vector_norms
        
        # 批量计算
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities
    
    def get_relevant_entities(self, question: str, k: int = 10) -> List[Dict]:
        """获取相关问题实体"""
        # 生成问题嵌入（使用BGE-M3）
        query_embedding = self.embedding_model.encode(question)
        
        # 批量计算相似度
        scores = self.batch_cosine_similarity(query_embedding)
        
        # 获取top-k索引
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果 - 返回所有top-k实体，不筛选degree
        relevant_entities = []
        for idx in top_indices:
            entity_id = self.entity_ids[idx]
            node = self.entity_map[entity_id]
            
            degree = node.get('degree', 0)
            if isinstance(degree, str):
                degree = int(degree) if degree.isdigit() else 0
            
            # 移除degree>0的条件，返回所有实体
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
        edge_map = {}  # 用于快速查找边
        
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
        
        print(f"种子实体: {entity_ids_set}")
        
        # 1. 直接连接
        print("\n【直接连接】")
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
                        print(f"  {source_node['name']} --[{edge['relation']}]--> {target_node['name']}")
        
        # 2. 2跳连接
        print("\n【2跳连接】")
        entity_list = list(entity_ids_set)
        for i, a in enumerate(entity_list):
            for b in entity_list[i+1:]:
                # 查找共同邻居
                neighbors_a = set(adjacency.get(a, []))
                neighbors_b = set(adjacency.get(b, []))
                common_neighbors = neighbors_a & neighbors_b
                
                for mid in common_neighbors:
                    # 找到路径 a -> mid -> b
                    print(f"  路径: {a} -> {mid} -> {b}")
                    
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
                            print(f"    {source_node['name']} --[{edge['relation']}]--> {target_node['name']}")
                    
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
                            print(f"    {source_node['name']} --[{edge['relation']}]--> {target_node['name']}")
        
        print(f"\n总共找到 {len(related_triplets)} 个相关三元组")
        return related_triplets
    
    def get_chunks_by_triplets(self, triplets: List[Dict]) -> List[Dict]:
        """根据三元组获取相关知识块"""
        chunk_ids = set()
        for triplet in triplets:
            if triplet.get('source_chunk'):
                chunk_ids.add(str(triplet['source_chunk']))  # 确保chunk_ids都是字符串
        
        chunks = []
        
        for chunk in self.chunks_data:
            # 将chunk的chunk_id也转换为字符串进行比较
            chunk_id_str = str(chunk['chunk_id'])
            
            if chunk_id_str in chunk_ids:
                chunks.append({
                    'chunk_id': chunk['chunk_id'],  # 保持原格式
                    'content': chunk['content'],
                    'source_paper': chunk['source_paper']
                })
        
        print(f"找到 {len(chunks)} 个匹配的chunk")
        return chunks
    
    def build_context(self, question: str) -> Dict[str, Any]:
        """
        构建并返回完整的上下文信息
        返回包含检索过程中所有中间结果的字典
        """
        print(f"\n{'='*60}")
        print(f"处理问题: {question}")
        print(f"{'='*60}")
        
        # 1. 获取相关实体
        print(f"\n【步骤1】检索相关实体 (k={self.config.get('k_entities', 10)})")
        relevant_entities = self.get_relevant_entities(question, k=self.config.get('k_entities', 10))
        
        print(f"找到 {len(relevant_entities)} 个相关实体:")
        for i, entity in enumerate(relevant_entities[:10], 1):  # 显示前10个
            print(f"  {i}. {entity['name']} (类型: {entity['type']}, 主题: {entity['topic']}, 相似度: {entity['score']:.4f})")
        if len(relevant_entities) > 10:
            print(f"  ... 等共 {len(relevant_entities)} 个实体")
        
        if not relevant_entities:
            print("未找到相关实体")
            return {
                'question': question,
                'entities': [],
                'triplets': [],
                'chunks': [],
                'context_text': "无相关医学文献"
            }
        
        # 2. 获取相关三元组
        print(f"\n【步骤2】检索相关三元组")
        entity_ids = [e['id'] for e in relevant_entities]
        triplets = self.get_related_triplets(entity_ids)
        
        print(f"找到 {len(triplets)} 个相关三元组:")
        for i, t in enumerate(triplets[:5], 1):  # 只显示前5个
            print(f"  {i}. {t['source']} --[{t['relation']}]--> {t['target']}")
        if len(triplets) > 5:
            print(f"  ... 等共 {len(triplets)} 个三元组")
        
        # 3. 获取相关知识块
        print(f"\n【步骤3】检索相关知识块")
        chunks = self.get_chunks_by_triplets(triplets)
        
        print(f"找到 {len(chunks)} 个相关知识块:")
        
        # 4. 构建文本块列表（格式化为清晰的列表）
        context_parts = []
        source_papers = set()
        
        for i, chunk in enumerate(chunks, 1):
            paper = chunk['source_paper']
            source_papers.add(f"文献{paper}")
            chunk_info = f"文本块{i}：【文献{paper}】 {chunk['content']}"
            context_parts.append(chunk_info)
            
            # 打印前3个文本块的内容摘要
            if i <= 3:
                print(f"\n  --- 文本块{i} (来源: 文献{paper}) ---")
                print(f"  {chunk['content'][:150]}..." if len(chunk['content']) > 150 else f"  {chunk['content']}")
        
        # 格式化参考文献字符串
        references = "\n".join(source_papers) if source_papers else "医学文献"
        
        # 构建完整的上下文文本
        if context_parts:
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "无相关医学文献"
            print("未找到相关知识块")
        
        # 打印最终构建的上下文
        print(f"\n【步骤4】构建的上下文 (共 {len(chunks)} 个文本块):")
        print(f"\n{context_text}")
        #print(f"\n参考文献: {references}")
        
        # 返回完整的检索结果
        return {
            'question': question,
            'entities': [{'name': e['name'], 'type': e['type'], 'topic': e['topic'], 'score': e['score']} 
                        for e in relevant_entities],
            'triplets': triplets,
            'chunks': chunks,
            'references': list(source_papers),
            'context_text': context_text
        }

def main():
    # 注意：这里不需要设置HF_TOKEN了，因为已经移除了LLM调用
    # 如果需要保持代码结构，可以留空或注释掉
    # HF_TOKEN = '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'
    # os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 初始化系统
    print("正在初始化KGRAG系统...")
    kg_rag = KGRAGSystem("config.json")
    
        # 测试多个问题
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    # 可以选择测试单个问题或所有问题
    question = "我肺癌术后稍微走两步就喘，还能运动干预吗？"  # 测试单个问题
    #question = questions["4"]  # 测试诊断标准
     
    # 构建并显示上下文
    result = kg_rag.build_context(question)
    
    print(f"\n{'='*60}")
    print(f"检索完成！")
    print(f"问题: {result['question']}")
    print(f"相关实体数量: {len(result['entities'])}")
    print(f"相关三元组数量: {len(result['triplets'])}")
    print(f"相关文本块数量: {len(result['chunks'])}")
    print(f"参考文献: {'\n '.join(result['references']) if result['references'] else '无'}")



if __name__ == "__main__":
    main()