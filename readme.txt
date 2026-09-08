安装依赖
python -m pip install -r requirements.txt

df <- read_excel("D:/0新时代/毕业论文/LLM测评/A最终数据.xlsx")

1 to 2
3个json

后端 to 前端？

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

无三元组的上下文
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
        
        # 4. 构建文本块列表（格式化为清晰的列表）
        context_parts = []
        for i, chunk in enumerate(chunks[:3], 1):  # 限制最多3个文本块
            context_parts.append(f"文本块{i}：【文献{chunk['source_paper']}】 {chunk['content']}")
        
        return "\n\n".join(context_parts)