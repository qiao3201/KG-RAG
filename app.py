from flask import Flask, render_template, request, jsonify
from kg_rag_system import KGRAGSystem
import os
import json

import sys
import io


# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

# 初始化KGRAG系统
HF_TOKEN = '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'
os.environ["HF_TOKEN"] = HF_TOKEN

print("正在初始化KGRAG系统...")
kg_rag = KGRAGSystem("config.json")
print("KGRAG系统初始化完成！")

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """处理查询请求"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '请输入问题'}), 400
        
        # 执行查询
        result = kg_rag.query(question)
        
        # 返回结果
        return jsonify({
            'success': True,
            'question': result['question'],
            'answer': result['answer'],
            'entities': result['entities'],
            'triplets': result['triplets'],
            'chunks': result['chunks'],
            'references': result['references']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)