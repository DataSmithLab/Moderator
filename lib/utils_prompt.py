from llama_cpp import Llama

def content_expand(
    content:str,
    expand_num:int
    ):
    query_prompt_template = "You act as an artistic Stable Diffusion prompt assistant. I have a content sentence: {content}, and I want to extend the content to prompts to input into the Stable Diffusion model. Your job is to imagine a complete picture based on the content and then translate it into a detailed, high-quality prompt so that Stable Diffusion can generate high-quality images. The generated prompt must start with [Generated Prompt]"
    query_prompt = query_prompt_template.format(content=content)
    output = llm("Q: "+query_prompt+" A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
    return output['choices'][0]['text']