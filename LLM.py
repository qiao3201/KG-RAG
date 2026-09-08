import json
import os
import sys
import io
from typing import Dict, Any
from huggingface_hub import InferenceClient

# 重新配置标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class LLMSystem:
    def __init__(self, config_path: str = "config.json"):
        """
        初始化 LLM 系统（无RAG）
        """
        self.load_config(config_path)
        self.init_llm()

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def init_llm(self):
        """初始化 Hugging Face Inference API 客户端"""
        if "HF_TOKEN" not in os.environ:
            raise ValueError("请设置环境变量 HF_TOKEN")
        
        self.llm_client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        self.llm_model = self.config.get('llm_model', "Qwen/Qwen3-4B-Instruct-2507:nscale")
        print(f"已连接到模型: {self.llm_model}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """执行单个查询"""
        
        # 构建提示词
        prompt = f"""你是一位专业的肿瘤护理专家，专门负责肺癌患者的癌因性疲乏（CRF）管理。请基于你所掌握的医学知识，回答以下患者问题。

【注意】
1.请确保回答专业、准确、清晰。
2.简要回答，字数在500内。

【患者问题】
{question}

请以结构化的方式回答，包括：
- 核心答案
- 注意事项
- 参考文献"""
        
        try:
            messages = [
                {"role": "system", "content": "你是一位专业的肿瘤护理专家，专门负责肺癌患者的癌因性疲乏（CRF）管理。"},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config.get('max_tokens', 500),
                temperature=self.config.get('temperature', 0.3)
            )
            
            response = completion.choices[0].message.content
            
        except Exception as e:
            print(f"调用API时出错: {e}")
            response = f"抱歉，生成回答时出现错误: {str(e)}"
        
        return {
            'question': question,
            'answer': response
        }


def main():
    # 设置HF_TOKEN
    HF_TOKEN = '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'
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 初始化系统
    llm_system = LLMSystem("config.json")
    
    # 加载问题文件
    with open('assessment_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    total_questions = 42
    repeat_times = 14
    
    # 重复执行14次
    for run in range(6, repeat_times + 1):
        output_file = f"LLM{run}.txt"
        print(f"\n{'='*60}")
        print(f"第 {run} 次运行，结果将保存到 {output_file}")
        print(f"{'='*60}")
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(f"LLM系统回答结果 - 第{run}次运行\n")
            out_file.write("包含：问题 + LLM回答\n")
            out_file.write(f"{'='*80}\n\n")
            
            # 循环处理问题1-42
            for i in range(1, total_questions + 1):
                question_id = str(i)
                if question_id in questions:
                    question = questions[question_id]
                    
                    print(f"  正在处理问题 {i}/{total_questions}...")
                    
                    # 执行查询
                    result = llm_system.query(question)
                    
                    # 输出格式：问题和LLM回答
                    out_file.write(f"【问题 {i}】\n")
                    out_file.write(f"{result['question']}\n\n")
                    
                    out_file.write("【LLM回答】\n")
                    out_file.write(f"{result['answer']}\n\n")
                    
                    # 添加分隔线
                    out_file.write(f"{'-'*60}\n\n")
                    out_file.flush()  # 确保立即写入
                    
                else:
                    out_file.write(f"【问题 {i}】\n")
                    out_file.write("未找到对应问题\n\n")
                    out_file.write(f"{'-'*60}\n\n")
            
            print(f"  第 {run} 次运行完成，已保存到 {output_file}")
    
    print(f"\n{'='*60}")
    print(f"全部完成！共执行 {repeat_times} 次")
    print(f"生成文件: LLM1.txt 到 LLM{repeat_times}.txt")
    print(f"每次处理 {total_questions} 个问题")

if __name__ == "__main__":
    main()