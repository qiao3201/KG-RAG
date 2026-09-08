import sys
import io
import json
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pyecharts import options as opts
from pyecharts.charts import Graph

# 加载知识图谱数据
with open('knowledge_graph_data.json', 'r', encoding='utf-8') as f:
    kg_data = json.load(f)

# 清洗节点名称，移除可能导致JS错误的特殊字符
def clean_name(name):
    """清洗节点名称，移除特殊字符并限制长度"""
    if not isinstance(name, str):
        name = str(name)
    # 移除换行符、制表符、多余空格
    name = name.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    name = re.sub(r'\s+', ' ', name)  # 多个空格合并为一个
    # 限制长度（避免过长）
    if len(name) > 25:
        name = name[:22] + '...'
    # 移除可能导致JS错误的特殊字符
    name = name.replace('"', '&quot;').replace("'", '&#39;')
    return name

# 建立 id 到 name 的映射
id_to_name = {entity['id']: clean_name(entity['name']) for entity in kg_data['nodes']}

# 计算节点的度（连接数）
degree = {}
for entity in kg_data['nodes']:
    source_count = sum(1 for e in kg_data['edges'] if e['source'] == entity['id'])
    target_count = sum(1 for e in kg_data['edges'] if e['target'] == entity['id'])
    degree[entity['id']] = source_count + target_count

# 筛选核心节点（度 ≥ 3，且不是 addition 类型）
core_node_ids = []
for entity in kg_data['nodes']:
    node_degree = degree.get(entity['id'], 0)
    # 筛选条件：度 >= 3，且类型不是 addition（addition 是长文本描述节点）
    if node_degree >= 3 and entity.get('type') != 'addition':
        core_node_ids.append(entity['id'])

# 获取核心节点的信息
core_nodes = [entity for entity in kg_data['nodes'] if entity['id'] in core_node_ids]

# 筛选边（只保留连接核心节点的边）
core_edges = []
for edge in kg_data['edges']:
    if edge['source'] in core_node_ids and edge['target'] in core_node_ids:
        core_edges.append(edge)

print(f"原始节点数: {len(kg_data['nodes'])}")
print(f"原始边数: {len(kg_data['edges'])}")
print(f"核心节点数: {len(core_nodes)}")
print(f"核心边数: {len(core_edges)}")

# 节点颜色配置
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

# 准备节点数据
nodes = []
for node in core_nodes:
    node_name = clean_name(node['name'])
    node_degree = degree.get(node['id'], 1)
    # 根据度设置节点大小
    symbol_size = min(50, max(15, node_degree * 3))
    node_type = node.get('type', 'symptom')
    nodes.append({
        "name": node_name,
        "symbolSize": symbol_size,
        "category": node_type,
        "value": node_degree,
        "itemStyle": {"color": color_map.get(node_type, '#CCCCCC')}
    })

# 准备边数据
links = []
for edge in core_edges:
    source_name = clean_name(id_to_name.get(edge['source'], str(edge['source'])))
    target_name = clean_name(id_to_name.get(edge['target'], str(edge['target'])))
    links.append({
        "source": source_name,
        "target": target_name,
        "label": edge.get('relation', '')
    })

# 如果没有足够的边，使用原始数据并降低筛选条件
if len(core_edges) < 10:
    print("警告：核心边数过少，降低筛选条件...")
    core_node_ids = []
    for entity in kg_data['nodes']:
        node_degree = degree.get(entity['id'], 0)
        if node_degree >= 2:
            core_node_ids.append(entity['id'])
    
    core_nodes = [entity for entity in kg_data['nodes'] if entity['id'] in core_node_ids]
    core_edges = []
    for edge in kg_data['edges']:
        if edge['source'] in core_node_ids and edge['target'] in core_node_ids:
            core_edges.append(edge)
    
    print(f"调整后核心节点数: {len(core_nodes)}")
    print(f"调整后核心边数: {len(core_edges)}")
    
    nodes = []
    for node in core_nodes:
        node_name = clean_name(node['name'])
        node_degree = degree.get(node['id'], 1)
        symbol_size = min(50, max(15, node_degree * 3))
        node_type = node.get('type', 'symptom')
        nodes.append({
            "name": node_name,
            "symbolSize": symbol_size,
            "category": node_type,
            "value": node_degree,
            "itemStyle": {"color": color_map.get(node_type, '#CCCCCC')}
        })
    
    links = []
    for edge in core_edges:
        source_name = clean_name(id_to_name.get(edge['source'], str(edge['source'])))
        target_name = clean_name(id_to_name.get(edge['target'], str(edge['target'])))
        links.append({
            "source": source_name,
            "target": target_name,
            "label": edge.get('relation', '')
        })

print(f"最终节点数: {len(nodes)}")
print(f"最终边数: {len(links)}")

# 生成交互式图谱（显示节点名称）
graph = (
    Graph(init_opts=opts.InitOpts(width="1400px", height="900px", theme="light", bg_color="#FFFFFF"))
    .add(
        series_name="知识图谱",
        nodes=nodes,
        links=links,
        layout="force",
        repulsion=500,
        edge_length=120,
        linestyle_opts=opts.LineStyleOpts(color="#999", width=1, type_="solid", curve=0.2),
        # 【修改点】将 is_show 改为 True 显示节点名称
        label_opts=opts.LabelOpts(
            is_show=True,           # 显示节点标签
            position="right",       # 标签位置：right, top, bottom, left
            font_size=10,           # 字体大小
            font_family="Microsoft YaHei",  # 字体
            color="#333333",        # 字体颜色
            formatter="{b}"         # 显示节点名称
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}<br/>类型: {c}<br/>连接数: {d}"
        ),
        categories=[
            {"name": "symptom", "itemStyle": {"color": "#E76F51"}},
            {"name": "tool", "itemStyle": {"color": "#F4A261"}},
            {"name": "therapy", "itemStyle": {"color": "#2A9D8F"}},
            {"name": "characteristic", "itemStyle": {"color": "#E9C46A"}},
            {"name": "indicator", "itemStyle": {"color": "#8ECAE6"}},
            {"name": "principle", "itemStyle": {"color": "#A8DADC"}},
            {"name": "factor", "itemStyle": {"color": "#B5838D"}},
            {"name": "addition", "itemStyle": {"color": "#9C89B8"}}
        ]
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
)

# 保存为HTML文件
output_file = "knowledge_graph_core.html"
graph.render(output_file)
print(f"知识图谱已保存为 {output_file}")