import ollama
import ast

class LLM:
    def __init__(
        self,
        model_name = "llama3.1:8b"
    ):
        self.model_name = model_name

    def query(
        self,
        system_prompt,
        user_prompt
    ):
        response = ollama.chat(
            model=self.model_name, 
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                }
            ]
        )
        result = response['message']['content']

        return self.remove_tags_content(text=result)
    
    def parse_llm_output_list( 
        self,
        output:str,
    ):
        try:
            # 尝试用ast.literal_eval解析字符串
            parsed = ast.literal_eval(output.strip())
            
            # 检查解析结果是否为字符串列表
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
            else:
                raise Exception(f"字符串解析后不是List[str]，而是: {type(parsed)}")
        except (SyntaxError, ValueError, TypeError) as e:
            raise Exception(f"无法将字符串解析为List[str]: {str(e)}")

    def remove_tags_content(self, text, start_tag='<think>', end_tag='</think>'):
        """
        删除文本中位于start_tag和end_tag之间的内容（包括标签本身）
        
        参数:
            text: 原始文本
            start_tag: 起始标签，默认为'<think>'
            end_tag: 结束标签，默认为'</RichMediaReference>'
        
        返回:
            处理后的文本
        """
        result = []
        i = 0
        while i < len(text):
            # 查找起始标签
            start_idx = text.find(start_tag, i)
            if start_idx == -1:
                # 没有找到起始标签，添加剩余文本并退出循环
                result.append(text[i:])
                break
            
            # 添加起始标签之前的内容
            result.append(text[i:start_idx])
            
            # 查找结束标签
            end_idx = text.find(end_tag, start_idx + len(start_tag))
            if end_idx == -1:
                # 没有找到结束标签，添加剩余文本并退出循环
                result.append(text[start_idx:])
                break
            
            # 跳过标签及其内容
            i = end_idx + len(end_tag)
        
        return ''.join(result)