"""
知识图谱构建模块
功能：加载向量化实体，读取关系数据，构建networkx图结构
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import os
from typing import Dict, List
from vectorize_entities import EntityVectorizer  # 导入向量化器


class MedicalKnowledgeGraph:
    """医学知识图谱构建器"""
    
    def __init__(self, vectorizer: EntityVectorizer):
        self.vectorizer = vectorizer
        self.graph = nx.Graph()  # 无向图
        self.entity_centrality = {}
        
        # 节点颜色映射（按类型）
        self.type_colors = {
            'symptom': '#FF6F61',      # 珊瑚红
            'therapy': '#6B5B95',      # 紫罗兰
            'factor': '#88B04B',       # 橄榄绿
            'indicator': '#F7CAC9',    # 淡粉色
            'tool': '#92A8D1',         # 淡蓝色
            'characteristic': '#955251', # 深红棕
            'principle': '#B565A7',    # 紫红色
            'addition': '#009B77',     # 青绿色
            'unknown': '#DDDDDD' 
        }
        
        # 节点大小映射（按类型）
        self.type_sizes = {
            'symptom': 500,      # 症状：最大
            'therapy': 400,      # 治疗方法
            'factor': 300,       # 因素
            'indicator': 300,    # 指标
            'tool': 250,         # 工具
            'characteristic': 200, # 特征
            'principle': 150,    # 原则
            'addition': 100      # 补充信息：最小
        }
    
    def load_relation_data(self) -> pd.DataFrame:
        """加载关系数据"""
        print("📂 加载关系数据...")
        
        relations_df = pd.read_excel('关系表.xlsx')
        relations_df.columns = ['id', 'head_entity', 'relation', 'tail_entity', 'source']
        
        # 确保ID为字符串
        relations_df['head_entity'] = relations_df['head_entity'].astype(str)
        relations_df['tail_entity'] = relations_df['tail_entity'].astype(str)
        
        print(f"  关系数量: {len(relations_df)}")
        print(f"  关系类型: {relations_df['relation'].unique().tolist()}")
        
        return relations_df
    
    def build_graph(self, relations_df: pd.DataFrame):
        """从关系数据构建知识图谱"""
        print("\n🔧 构建知识图谱...")
        # 1. 创建名称到ID的映射字典
        name_to_id = {}
        for entity_id, details in self.vectorizer.entity_details.items():
            entity_name = details['name']
            name_to_id[entity_name] = entity_id
        
        print(f"  创建了 {len(name_to_id)} 个名称到ID的映射")
        
        # 1. 添加所有实体作为节点
        for entity_id, details in self.vectorizer.entity_details.items():
            if entity_id in self.vectorizer.entity_vectors:
                self.graph.add_node(
                    entity_id,
                    name=details['name'],
                    type=details['type'],
                    topic=details['topic'],
                    id_prefix=details['id_prefix'],
                    vector=self.vectorizer.entity_vectors[entity_id],
                    size=self.type_sizes.get(details['type'], 200),
                    color=self.type_colors.get(details['type'], '#6c757d')
                )
        
        # 2. 添加关系作为边
        edge_count = 0
        missing_entities = set()
        for _, row in relations_df.iterrows():
            head_name = str(row['head_entity']).strip()
            tail_name = str(row['tail_entity']).strip()
            relation_type = row['relation']
 
            # 通过名称查找ID
            head_id = name_to_id.get(head_name)
            tail_id = name_to_id.get(tail_name)
            
            # 确保两个实体都存在
            if head_id in self.graph and tail_id in self.graph:
                self.graph.add_edge(
                    head_id,
                    tail_id,
                    relation=relation_type,
                    label=relation_type,
                    weight=1.0
                )
                edge_count += 1
        
        print(f"  成功添加边数: {edge_count}")
    
    def calculate_metrics(self):
        """计算图谱指标"""
        print("\n📊 计算图谱指标...")
        
        if len(self.graph.nodes()) == 0:
            print("⚠️  图为空，无法计算指标")
            return
        
        # 基本统计
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        
        print(f"  节点数: {node_count}")
        print(f"  边数: {edge_count}")
        print(f"  密度: {nx.density(self.graph):.4f}")
        
        # 度中心性
        self.entity_centrality = nx.degree_centrality(self.graph)
        
        # 连通分量
        components = list(nx.connected_components(self.graph))
        print(f"  连通分量: {len(components)} 个")
        
        if components:
            largest_component = max(components, key=len)
            print(f"  最大连通分量: {len(largest_component)} 个节点")
        
        # 按类型统计
        type_counts = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        print(f"\n📋 节点类型分布:")
        for node_type, count in sorted(type_counts.items()):
            type_zh = self.vectorizer._get_type_name(node_type)
            print(f"  {type_zh:10s}: {count:3d} 个")
    
    def get_top_entities(self, top_n: int = 10):
        """获取最重要的实体（按连接数）"""
        if not self.entity_centrality:
            print("⚠️  未计算中心性指标")
            return []
        
        sorted_entities = sorted(
            self.entity_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        print(f"\n🏆 最重要的 {top_n} 个实体:")
        for rank, (entity_id, centrality) in enumerate(sorted_entities, 1):
            entity_name = self.graph.nodes[entity_id].get('name', entity_id)
            degree = self.graph.degree(entity_id)
            print(f"  {rank:2d}. {entity_name:20s} 连接数:{degree:3d} 中心性:{centrality:.3f}")
        
        return sorted_entities
    
    def save_graph_data(self, filepath: str = 'knowledge_graph_data.json'):
        """保存图数据（供可视化使用）"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'metrics': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if len(self.graph.nodes()) > 0 else 0
            }
        }
        
        # 保存节点
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node_id,
                'name': node_data.get('name', ''),
                'type': node_data.get('type', 'unknown'),
                'topic': node_data.get('topic', ''),
                'degree': self.graph.degree(node_id)
            })
        
        # 保存边
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': source,
                'target': target,
                'relation': edge_data.get('relation', 'unknown')
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 图数据已保存到: {filepath}")
        return filepath


def main():
    """主函数：构建知识图谱"""
    print("=" * 60)
    print("🏥 癌因性疲乏知识图谱 - 图谱构建")
    print("=" * 60)
    
    # 1. 初始化向量化器
    vectorizer = EntityVectorizer("dummy_token")  # Token仅用于加载
    vectors_file = 'medical_kg_vectors.json'
    
    if not os.path.exists(vectors_file):
        print(f"❌ 向量文件不存在: {vectors_file}")
        print("请先运行 1_vectorize_entities.py")
        return
    
    # 2. 加载向量
    print("📂 加载向量文件...")
    if not vectorizer.load_vectors(vectors_file):
        return
    
    # 3. 初始化知识图谱
    kg = MedicalKnowledgeGraph(vectorizer)
    
    # 4. 加载关系数据
    relations_df = kg.load_relation_data()
    
    # 5. 构建图谱
    kg.build_graph(relations_df)
    
    # 6. 计算指标
    kg.calculate_metrics()
    
    # 7. 显示重要实体
    kg.get_top_entities(15)
    
    # 8. 保存图数据
    kg.save_graph_data()
    
    print("\n✅ 知识图谱构建完成!")
    print("   下一步: 运行 3_visualize_graph.py")


if __name__ == "__main__":
    main() 