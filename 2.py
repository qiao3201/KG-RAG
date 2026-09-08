# 检查 PyEcharts 版本和依赖
import pyecharts
print(f"PyEcharts 版本: {pyecharts.__version__}")

# 生成一个简单的测试图谱
from pyecharts.charts import Graph
from pyecharts import options as opts

# 简单测试
test_nodes = [
    {"name": "节点1", "symbolSize": 30},
    {"name": "节点2", "symbolSize": 30},
    {"name": "节点3", "symbolSize": 30}
]
test_links = [
    {"source": "节点1", "target": "节点2"},
    {"source": "节点2", "target": "节点3"}
]

test_graph = (
    Graph()
    .add("测试", test_nodes, test_links, layout="force")
    .set_global_opts(title_opts=opts.TitleOpts(title="测试图谱"))
)
test_graph.render("test_graph.html")
print("测试图谱已生成，请打开 test_graph.html 查看")