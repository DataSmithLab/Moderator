import ollama

class LLM:
    def __init__(
        self,
        model_name = "llama3.1:8b"
    ):
        self.model_name = model_name
        assert self.model_name in ollama.model_list()

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