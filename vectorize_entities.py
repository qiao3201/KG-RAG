"""
实体向量化模块
功能：读取Excel数据，调用BGE-M3 API向量化所有实体，保存到JSON文件
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from typing import Dict, List

class EntityVectorizer:
    """实体向量化处理器"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        
        # 存储结构
        self.entity_vectors = {}  # {entity_id: vector}
        self.entity_details = {}  # {entity_id: {name, type, id_prefix, topic}}
        
        # 主题映射（根据ID第一位）
        self.topic_mapping = {
            '0': '核心症状',
            '1': '疾病特征',
            '2': '评估工具',
            '3': '运动干预',
            '4': '营养管理',
            '5': '心理干预',
            '6': '睡眠管理',
            '7': '物理疗法'
        }
        
        # 类型映射
        self.type_mapping = {
            'characteristic': '疾病特征',
            'factor': '影响因素',
            'tool': '评估工具',
            'therapy': '治疗方法',
            'principle': '干预原则',
            'symptom': '症状',
            'indicator': '临床指标',
            'addition': '补充信息'
        }
    
    def _extract_id_prefix(self, entity_id: str) -> str:
        """提取ID前缀（主题信息）"""
        if pd.isna(entity_id) or not str(entity_id).strip():
            return '0'
        entity_str = str(entity_id).strip()
        return entity_str[0] if entity_str else '0'
    
    def _get_topic_name(self, id_prefix: str) -> str:
        """获取主题名称"""
        return self.topic_mapping.get(id_prefix, f"主题{id_prefix}")
    
    def _get_type_name(self, entity_type: str) -> str:
        """获取类型中文名"""
        return self.type_mapping.get(entity_type, entity_type)
    
    def enrich_entity_text(self, entity_id: str, entity_name: str, entity_type: str) -> str:
        """为实体名称添加上下文，增强语义"""
        id_prefix = self._extract_id_prefix(entity_id)
        topic_name = self._get_topic_name(id_prefix)
        type_name = self._get_type_name(entity_type)
        
        # 构建增强文本
        return f"{topic_name}领域中的{type_name}：{entity_name}（医学知识ID:{entity_id}）"
    
    def vectorize_entity(self, entity_id: str, entity_name: str, entity_type: str) -> np.ndarray:
        """向量化单个实体"""
        id_prefix = self._extract_id_prefix(entity_id)
        
        # 检查是否已向量化
        if entity_id in self.entity_vectors:
            print(f"⏭️  已缓存: {entity_name}")
            return self.entity_vectors[entity_id]
        
        # 构建增强文本
        enhanced_text = self.enrich_entity_text(entity_id, entity_name, entity_type)
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": enhanced_text},
                timeout=30
            )
            
            if response.status_code == 200:
                vector = np.array(response.json())
                
                # 存储结果
                self.entity_vectors[entity_id] = vector
                self.entity_details[entity_id] = {
                    'name': entity_name,
                    'type': entity_type,
                    'id_prefix': id_prefix,
                    'topic': self._get_topic_name(id_prefix),
                    'type_zh': self._get_type_name(entity_type)
                }
                
                topic_name = self._get_topic_name(id_prefix)
                print(f"✅ [{id_prefix}]{topic_name}: {entity_name}")
                return vector
                
            elif response.status_code == 503:
                print(f"⏳ {entity_name}: 模型加载中，等待10秒...")
                time.sleep(10)
                return self.vectorize_entity(entity_id, entity_name, entity_type)
                
            else:
                print(f"❌ {entity_name}: API错误 {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏱️  {entity_name}: 请求超时")
            return None
        except Exception as e:
            print(f"⚠️  {entity_name}: {e}")
            return None
    
    def batch_vectorize(self, entities_data: pd.DataFrame, batch_delay: float = 0.3) -> Dict[str, np.ndarray]:
        """批量向量化所有实体"""
        total_entities = len(entities_data)
        print(f"\n🎯 开始批量向量化 {total_entities} 个实体...")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        
        for idx, row in entities_data.iterrows():
            entity_id = str(row['entity_id'])
            entity_name = row['entity']
            entity_type = row['type']
            
            # 进度显示
            if idx % 10 == 0:
                print(f"📈 进度: {idx+1}/{total_entities} ({((idx+1)/total_entities*100):.1f}%)")
            
            result = self.vectorize_entity(entity_id, entity_name, entity_type)
            
            if result is not None:
                success_count += 1
            else:
                fail_count += 1
            
            # 请求间隔
            time.sleep(batch_delay)
        
        print("\n" + "=" * 60)
        print(f"📊 向量化完成!")
        print(f"  成功: {success_count} 个实体")
        print(f"  失败: {fail_count} 个实体")
        
        # 按主题统计
        print(f"\n📋 按主题分布:")
        topic_stats = {}
        for details in self.entity_details.values():
            topic = details['topic']
            topic_stats[topic] = topic_stats.get(topic, 0) + 1
        
        for topic, count in sorted(topic_stats.items()):
            print(f"  {topic:15s}: {count:3d} 个实体")
        
        return self.entity_vectors
    
    def save_vectors(self, filepath: str = 'medical_kg_vectors.json'):
        """保存向量到文件"""
        # 转换numpy数组为列表
        vectors_serializable = {}
        for entity_id, vector in self.entity_vectors.items():
            if isinstance(vector, np.ndarray):
                vectors_serializable[entity_id] = vector.tolist()
            else:
                vectors_serializable[entity_id] = vector
        
        data = {
            'entity_vectors': vectors_serializable,
            'entity_details': self.entity_details,
            'metadata': {
                'total_entities': len(self.entity_vectors),
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model': 'BGE-M3'
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 向量数据已保存到: {filepath}")
        print(f"   包含 {len(self.entity_vectors)} 个实体向量")
        return filepath
    
    def load_vectors(self, filepath: str = 'medical_kg_vectors.json') -> bool:
        """从文件加载向量"""
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载向量
        self.entity_vectors = {}
        for entity_id, vector_list in data['entity_vectors'].items():
            self.entity_vectors[entity_id] = np.array(vector_list)
        
        # 加载详情
        self.entity_details = data['entity_details']
        
        metadata = data.get('metadata', {})
        print(f"📂 已加载向量文件: {filepath}")
        print(f"   实体数量: {len(self.entity_vectors)}")
        print(f"   创建时间: {metadata.get('created_at', '未知')}")
        
        return True


def load_entity_data():
    """加载实体数据"""
    print("📂 加载实体数据...")
    
    # 加载实体数据
    entities_df = pd.read_excel('实体表.xlsx', sheet_name=0)
    attributes_df = pd.read_excel('实体表.xlsx', sheet_name=1)
    
    # 统一列名
    entities_df.columns = ['entity_id', 'entity', 'type']
    attributes_df.columns = ['entity_id', 'entity', 'type']
    
    # 合并所有实体
    all_entities = pd.concat([
        entities_df[['entity_id', 'entity', 'type']],
        attributes_df[['entity_id', 'entity', 'type']]
    ], ignore_index=True)
    
    # 确保entity_id为字符串
    all_entities['entity_id'] = all_entities['entity_id'].astype(str)
    
    print(f"📊 数据统计:")
    print(f"  主实体: {len(entities_df)} 个")
    print(f"  属性实体: {len(attributes_df)} 个")
    print(f"  总计: {len(all_entities)} 个实体")
    
    return all_entities


def main():
    """主函数：向量化实体"""
    print("=" * 60)
    print("🏥 癌因性疲乏知识图谱 - 实体向量化")
    print("=" * 60)
    
    # 1. 设置API Token
    #HF_TOKEN = input("请输入HuggingFace Token (或直接粘贴): ").strip()
    HF_TOKEN = "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"
    if not HF_TOKEN.startswith('hf_'):
        print("⚠️  Token格式不正确，应以'hf_'开头")
        return
    
    # 2. 加载数据
    entities_data = load_entity_data()
    
    # 3. 初始化向量化器
    vectorizer = EntityVectorizer(HF_TOKEN)
    
    # 4. 批量向量化
    print("\n" + "=" * 60)
    vectorizer.batch_vectorize(entities_data)
    
    # 5. 保存结果
    print("\n" + "=" * 60)
    output_file = vectorizer.save_vectors()
    
    print("\n✅ 实体向量化完成!")
    print(f"   输出文件: {output_file}")
    print("   下一步: 运行 2_build_knowledge_graph.py")


if __name__ == "__main__":   
    main()