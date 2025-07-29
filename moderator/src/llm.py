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
        return result
    
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
