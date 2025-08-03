from moderator.src.llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import copy
from moderator.src.context_desc import ContextDesc
from typing import List

class QueryExpander:
    def __init__(
        self,
        model_name = "llama3.1:8b",
        prompts_folder_path = "moderator/prompts",
    ):
        self.model_name = model_name
        self.prompts_folder_path = prompts_folder_path
        self.load_prompt()
        self.llm = LLM(
            model_name=self.model_name
        )

    def load_prompt(
        self
    ):
        with open(f"{self.prompts_folder_path}/vocabulary_expand-system.prompt", "r") as f:
            self.vocabulary_expand_system_prompt = f.read()
            f.close()
        with open(f"{self.prompts_folder_path}/vocabulary_expand-subconcepts.prompt", "r") as f:
            self.vocabulary_expand_subconcepts_prompt = f.read()
            f.close()
        with open(f"{self.prompts_folder_path}/vocabulary_expand-synonyms.prompt", "r") as f:
            self.vocabulary_expand_synonyms_prompt = f.read()
            f.close()
        with open(f"{self.prompts_folder_path}/vocabulary_expand-description.prompt", "r") as f:
            self.vocabulary_expand_description_prompt = f.read()
            f.close()
        with open(f"{self.prompts_folder_path}/blank_expand.prompt", "r") as f:
            self.blank_expand_prompt = f.read()
            f.close()
        with open(f"{self.prompts_folder_path}/prompt_expand.prompt", "r") as f:
            self.prompt_expand_prompt = f.read()
            f.close()
            
    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def vocabulary_expand(
        self,
        vocabulary:str,
        type:str="synonyms",
        expand_num=4
    ):
        if type=="synonyms":
            prompt = self.vocabulary_expand_synonyms_prompt.format(
                vocabulary=vocabulary,
                expand_num=expand_num
            )
        elif type == "subconcepts":
            prompt = self.vocabulary_expand_subconcepts_prompt.format(
                vocabulary=vocabulary,
                expand_num=expand_num
            )
        elif type == "description":
            prompt = self.vocabulary_expand_description_prompt.format(
                vocabulary=vocabulary,
                expand_num=expand_num
            )
        else:
            assert type in ["synonyms", "subconcepts", "description"]
            raise AssertionError

        response = self.llm.query(
            system_prompt=self.vocabulary_expand_system_prompt,
            user_prompt=prompt
        )
        return self.llm.parse_llm_output_list(response)

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def blank_expansion2list(
        self,
        expand_num=30,
        expand_key="obj"
    ):
        prompt = self.blank_expand_prompt.format(
            expand_key=expand_key,
            expand_num=expand_num
        )
        response = self.llm.query(
            system_prompt=None,
            user_prompt=prompt
        )
        return self.llm.parse_llm_output_list(response)
    
    def blank_expansion(
            self,
            context_desc:ContextDesc,
            expand_num=30
    ):
        expanded_context_desc_list = [context_desc]*expand_num
        blank_contexts = context_desc.get_blank_contexts()
        for context_key in blank_contexts:
            context_list = self.blank_expansion2list(
                expand_num=expand_num, 
                expand_key=context_key
            )
            context_list = context_list*expand_num
            for i in range(0, expand_num):
                expanded_context_desc_list[i].set_context(
                    context_key=context_key,
                    context_value=context_list[i]
                )
        return expanded_context_desc_list

    def content_expansion(
        self,
        context_desc: ContextDesc,
        nonblank_expand_num = 4,
        nonblank_expand_key = "obj",
        nonblank_expand_type = "synonyms",
        blank_expand_num = 30,
        swap_value=None
    ):
        expanded_context_desc_list = self.blank_expansion(
            context_desc=context_desc,
            expand_num=blank_expand_num
        )
        
        original_non_blank_context = context_desc.get_context(nonblank_expand_key)
        word_list = self.vocabulary_expand(
            vocabulary=original_non_blank_context,
            type=nonblank_expand_type,
            expand_num=nonblank_expand_num
        )

        final_context_list = []
        for expanded_context_desc in expanded_context_desc_list:
            for word in [original_non_blank_context]+word_list:
                temp_expanded_context_desc = copy.deepcopy(expanded_context_desc)
                temp_expanded_context_desc.set_context(
                    context_key=nonblank_expand_key,
                    context_value=word
                )
                final_context_list.append(
                    temp_expanded_context_desc
                )
        if swap_value is None:
            return final_context_list
        else:
            swap_final_context_list = []
            for expanded_context_desc in expanded_context_desc_list:
                for word in ( len(word_list)+1 )*[swap_value]:
                    temp_expanded_context_desc = copy.deepcopy(expanded_context_desc)
                    temp_expanded_context_desc.set_context(
                        context_key=nonblank_expand_key,
                        context_value=word
                    )
                    swap_final_context_list.append(
                        temp_expanded_context_desc
                    )
            return final_context_list, swap_final_context_list

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def prompt_expansion(
            self,
            context_list: List[ContextDesc]
    ):
        prompt_list = []
        for context in context_list:
            context: ContextDesc
            expand_prompt = self.llm.query(
                user_prompt=self.prompt_expand_prompt.format(
                    obj=context.get_context("obj"),
                    sty=context.get_context("sty"),
                    act=context.get_context("act")
                )
            )
            prompt_list.append(
                expand_prompt
            )
        return prompt_list

    def prompt_concate(
            self,
            context_list: List[ContextDesc]
    ):
        prompt_list = []
        for context in context_list:
            context: ContextDesc
            prompt = "object:{obj}, style:{sty}, action:{act}".format(
                obj=context.get_context("obj"),
                sty=context.get_context("sty"),
                act=context.get_context("act")
            )
            prompt_list.append(
                prompt
            )
        return prompt_list

    def overall_expansion(
        self,
        input_context_desc:ContextDesc,
        swap_context_desc:ContextDesc=None,
        expand_key="obj",
        expand_type = "synonyms",
        prompt_expand=False
    ):
        if swap_context_desc is None:
            context_list = self.content_expansion(
                context_desc=input_context_desc,
                nonblank_expand_num = 4,
                nonblank_expand_key = expand_key,
                nonblank_expand_type = expand_type,
                blank_expand_num= 30
            )
            if prompt_expand:
                prompt_list = self.prompt_expansion(context_list=context_list)
            else:
                prompt_list = self.prompt_concate(context_list=context_list)
            return prompt_list
        else:
            context_list, swap_context_list = self.content_expansion(
                context_desc=input_context_desc,
                nonblank_expand_num = 4,
                nonblank_expand_key = expand_key,
                nonblank_expand_type = expand_type,
                blank_expand_num=30,
                swap_value=swap_context_desc.get_context(expand_key)
            )
            if prompt_expand:
                real_prompt_list = self.prompt_expansion(context_list=context_list)
                swap_prompt_list = self.prompt_expansion(context_list=swap_context_list)
            else:
                real_prompt_list = self.prompt_concate(context_list=context_list)
                swap_prompt_list = self.prompt_concate(context_list=swap_context_list)
            return real_prompt_list, swap_prompt_list