import json
import os
import sys
import io
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
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
    
    def init_llm(self):
        """初始化 Hugging Face Inference API 客户端 - 修复版本"""
        if "HF_TOKEN" not in os.environ:
            raise ValueError("请设置环境变量 HF_TOKEN")
        
        # 获取模型名称（从config中读取，如果没有则使用默认值）
        self.llm_model = self.config.get('llm_model',  "Qwen/Qwen3-4B-Instruct-2507:nscale")
        
        # 初始化客户端（不在这里指定模型）
        self.llm_client = InferenceClient(
            api_key=os.environ["HF_TOKEN"]
        )
        print(f"Hugging Face 客户端已初始化，使用模型: {self.llm_model}")
    
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
            print("无chunk")
            return "无相关医学文献"
        
        # 4. 构建文本块列表 - 移除数量限制，返回所有chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):  # 不再限制数量
            context_parts.append(f"文本块{i}：【文献{chunk['source_paper']}】 {chunk['content']}")
        
        return "\n\n".join(context_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询 - 修复LLM调用部分"""
        # 构建上下文
        context = self.build_context(question)
        
        # 获取相关实体
        relevant_entities = self.get_relevant_entities(question, k=self.config.get('k_entities', 10))
        
        triplets = []  # 初始化三元组列表
        chunks = []  # 初始化chunks列表
        source_papers = set()
        
        if relevant_entities:
            entity_ids = [e['id'] for e in relevant_entities]
            triplets = self.get_related_triplets(entity_ids)  # 获取三元组
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
            # 修复：使用 text_generation 方法（更稳定）
            response = self.llm_client.text_generation(
                prompt=prompt,
                model=self.llm_model,
                max_new_tokens=self.config.get('max_tokens', 500),
                temperature=self.config.get('temperature', 0.1),
                return_full_text=False  # 只返回生成的部分，不包含原始prompt
            )
            
            # 处理返回结果
            if isinstance(response, str):
                answer = response
            elif isinstance(response, dict) and 'generated_text' in response:
                answer = response['generated_text']
            else:
                answer = str(response)
            
        except Exception as e:
            print(f"调用API时出错: {e}")
            # 尝试备用方法：使用 chat_completion
            try:
                messages = [
                    {"role": "system", "content": "你是一位专业的肿瘤护理专家，专门负责肺癌患者的癌因性疲乏（CRF）管理。"},
                    {"role": "user", "content": prompt}
                ]
                
                completion = self.llm_client.chat_completion(
                    messages=messages,
                    model=self.llm_model,
                    max_tokens=self.config.get('max_tokens', 500),
                    temperature=self.config.get('temperature', 0.1)
                )
                
                answer = completion.choices[0].message.content
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                answer = f"抱歉，生成回答时出现错误: {str(e)}"
        
        # 返回结果（现在包含triplets）
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'triplets': triplets,  # 添加三元组
            'chunks': chunks,      # 添加chunks
            'references': list(source_papers),
            'entities': relevant_entities  # 添加实体
        }

def main():
    # 请在这里设置您的 HF_TOKEN（建议从环境变量读取，不要硬编码）
    # 方式1：从环境变量读取（推荐）
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # 方式2：如果没有设置环境变量，可以在这里手动设置（不推荐，但测试用）
    if not HF_TOKEN:
        HF_TOKEN = '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'  # 请替换为您的有效token
    
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 初始化系统
    kg_rag = KGRAGSystem("config.json")
    
    # 加载问题文件
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 打开输出文件 - 修改为test14.txt
    with open('test14.txt', 'w', encoding='utf-8') as out_file:
        out_file.write("KGRAG系统检索结果\n")
        out_file.write("包含：问题 + 实体 + 三元组 + 文本块 + LLM回答\n")
        out_file.write(f"{'='*80}\n\n")
        
        # 循环处理问题1-42
        total_questions = 1
        for i in range(1, total_questions + 1):
            question_id = str(i)
            if question_id in questions:
                question = questions[question_id]
                
                print(f"\n正在处理问题 {i}/{total_questions}: {question[:50]}...")
                
                # 执行查询
                result = kg_rag.query(question)
                
                # 输出格式：包含所有信息
                out_file.write(f"【问题 {i}】\n")
                out_file.write(f"{result['question']}\n\n")
                
                # 实体部分
                out_file.write("【相关实体】\n")
                if result['entities']:
                    for j, entity in enumerate(result['entities'], 1):
                        out_file.write(f"  {j}. {entity['name']} (类型: {entity['type']}, 主题: {entity['topic']}, 相似度: {entity['score']:.4f})\n")
                else:
                    out_file.write("  未检索到相关实体\n")
                out_file.write("\n")
                
                # 三元组部分
                out_file.write("【相关三元组】\n")
                if result['triplets']:
                    for j, triplet in enumerate(result['triplets'], 1):
                        out_file.write(f"  {j}. {triplet['source']} --[{triplet['relation']}]--> {triplet['target']}\n")
                else:
                    out_file.write("  未检索到相关三元组\n")
                out_file.write("\n")
                
                # 文本块部分 - 移除数量限制，输出所有chunks
                out_file.write("【相关文本块】\n")
                if result['chunks']:
                    for j, chunk in enumerate(result['chunks'], 1):  # 不再限制数量
                        out_file.write(f"  文本块{j}：【文献{chunk['source_paper']}】 {chunk['content']}\n\n")
                else:
                    out_file.write("  未检索到相关文本块\n\n")
                
                # LLM回答
                out_file.write("【LLM回答】\n")
                out_file.write(f"{result['answer']}\n\n")
                
                # 参考文献
                out_file.write("【参考文献】\n")
                if result['references']:
                    for ref in result['references']:
                        out_file.write(f"  {ref}\n")
                else:
                    out_file.write("  无参考文献\n")
                
                # 添加分隔线
                out_file.write(f"\n{'-'*60}\n\n")
                out_file.flush()  # 确保立即写入
                
                print(f"  完成 - 实体: {len(result['entities'])}, 三元组: {len(result['triplets'])}, 文本块: {len(result['chunks'])}")
            else:
                error_msg = f"警告: 问题 ID {question_id} 不存在\n"
                print(error_msg)
                out_file.write(f"【问题 {question_id}】\n")
                out_file.write("未找到对应问题\n\n")
                out_file.write(f"{'-'*60}\n\n")
    
    print(f"\n{'='*60}")
    print(f"处理完成！结果已保存到 test14.txt")
    print(f"共处理了 {total_questions} 个问题")

if __name__ == "__main__":
    main()