from moderator.src.query_expander import QueryExpander
from moderator.src.context_desc import ContextDesc

query_expander = QueryExpander(
    model_name="qwen3:8b"
)

context_desc = ContextDesc(
    obj="cat"
)

swap_context_desc = ContextDesc(
    obj="dog"
)

prompt_list = query_expander.overall_expansion(
    input_context_desc=context_desc,
    swap_context_desc=swap_context_desc,
    expand_key="obj",
    expand_type = "description",
    prompt_expand=False
)
print(prompt_list)