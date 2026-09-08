import sys
import io

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import networkx as nx
import json

# 加载知识图谱数据
with open('knowledge_graph_data.json', 'r', encoding='utf-8') as f:
    kg_data = json.load(f)

# 创建有向图
G = nx.DiGraph()

# 添加实体节点（使用 name 作为节点标识）
for entity in kg_data['nodes']:
    G.add_node(entity['name'], 
               id=entity['id'],
               type=entity['type'], 
               topic=entity['topic'],
               degree=entity.get('degree', 0))

# 添加三元组边（edges 中使用 source 和 target）
for edge in kg_data['edges']:
    # 需要根据节点名称找到对应的节点
    # 由于 nodes 中有 name 字段，但 edges 中使用的是 id 作为 source/target
    # 需要建立 id 到 name 的映射
    pass

# 建立 id 到 name 的映射
id_to_name = {entity['id']: entity['name'] for entity in kg_data['nodes']}

# 添加边（使用映射后的节点名称）
for edge in kg_data['edges']:
    source_name = id_to_name.get(edge['source'], edge['source'])
    target_name = id_to_name.get(edge['target'], edge['target'])
    G.add_edge(source_name, target_name, relation=edge.get('relation', ''))

# 按实体类型设置节点颜色
color_map = {
    'symptom': '#E76F51',      # 症状类 - 橙色
    'tool': '#F4A261',         # 工具类 - 沙色
    'therapy': '#2A9D8F',      # 治疗类 - 绿色
    'characteristic': '#E9C46A', # 特征类 - 金黄色
    'indicator': '#8ECAE6',    # 指标类 - 浅蓝色
    'principle': '#A8DADC',    # 原则类 - 浅青色
    'factor': '#B5838D',       # 因素类 - 玫瑰色
    'addition': '#9C89B8'      # 补充类 - 淡紫色
}

# 获取节点颜色
node_colors = []
for node in G.nodes:
    node_type = G.nodes[node].get('type', 'symptom')
    node_colors.append(color_map.get(node_type, '#CCCCCC'))

# 按度中心性设置节点大小
degree = dict(G.degree)
# 设置节点大小范围，避免过大或过小
node_sizes = [min(80, max(20, degree[node] * 15)) for node in G.nodes]

from pyecharts import options as opts
from pyecharts.charts import Graph

# 准备节点数据
nodes = []
for node in G.nodes:
    node_degree = degree.get(node, 1)
    node_type = G.nodes[node].get('type', 'symptom')
    nodes.append({
        "name": node,
        "symbolSize": min(80, max(20, node_degree * 12)),
        "category": node_type,
        "itemStyle": {"color": color_map.get(node_type, '#CCCCCC')}
    })

# 准备边数据
links = []
for edge in G.edges:
    links.append({
        "source": edge[0],
        "target": edge[1],
        "label": G.edges[edge].get('relation', '')
    })

# 生成交互式图谱
graph = (
    Graph(init_opts=opts.InitOpts(width="1400px", height="900px", theme="light"))
    .add(
        series_name="知识图谱",
        nodes=nodes,
        links=links,
        layout="force",
        repulsion=500,
        edge_length=150,
        linestyle_opts=opts.LineStyleOpts(color="#999", width=1, type_="solid", curve=0.3),
        label_opts=opts.LabelOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}<br/>类型: {c}<br/>连接数: {d}"
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="肺癌癌因性疲乏知识图谱可视化",
            subtitle=f"节点数: {len(nodes)} | 关系数: {len(links)}",
            pos_left="center"
        ),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        legend_opts=opts.LegendOpts(
            is_show=True,
            pos_top="bottom",
            pos_left="center"
        )
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=False)
    )
)

# 保存为HTML文件
graph.render("knowledge_graph.html")
print(f"知识图谱可视化已生成，共 {len(nodes)} 个节点，{len(links)} 条关系")
print("文件已保存为 knowledge_graph.html")