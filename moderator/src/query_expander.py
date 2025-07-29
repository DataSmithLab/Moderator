from moderator.src.llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        context_desc:dict,
        expand_num=30,
        expand_key="obj"
    ):
        prompt = self.blank_expand_prompt.format(
            expand_key=expand_key,
            expand_num=expand_num
        )
        response = self.llm.query(
            user_prompt=prompt
        )
        return self.llm.parse_llm_output_list(response)
    
    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def blank_expansion(
            self,
            context_desc:dict,
            expand_num=30
    ):
        expanded_context_desc_list = [context_desc]*expand_num
        for context_key, context_value in context_desc.items():
            if context_value is None or context_value == "":
                context_list = self.blank_expansion2list(
                    context_desc, expand_num, expand_key=context_key
                )
                # TODO
                context_list = context_list*10
                print(context_key, context_list)
                for i in range(0, expand_num):
                    expanded_context_desc_list[i][context_key] = context_list[i]
        return expanded_context_desc_list

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def content_expansion(
        self,
        context_desc:dict,
        expand_num_1 = 4,
        expand_key_1 = "obj",
        expand_1_type = "synonyms",
        expand_num_2 = 30,
        swap_value_1=None
    ):
        expanded_context_desc_list = self.blank_expansion(
            context_desc=context_desc,
            expand_num=expand_num_2
        )
        word_list = self.vocabulary_expand(
            vocabulary=context_desc[expand_key_1],
            type=expand_1_type,
            expand_num=expand_num_1
        )
        final_context_list = []
        for expanded_context_desc in expanded_context_desc_list:
            for word in [context_desc[expand_key_1]]+word_list:
                temp_expanded_context_desc = copy.deepcopy(expanded_context_desc)
                temp_expanded_context_desc[expand_key_1] = word
                final_context_list.append(
                    temp_expanded_context_desc
                )
        if swap_value_1 is None:
            return final_context_list
        else:
            swap_final_context_list = []
            for expanded_context_desc in expanded_context_desc_list:
                for word in ( len(word_list)+1 )*[swap_value_1]:
                    temp_expanded_context_desc = copy.deepcopy(expanded_context_desc)
                    temp_expanded_context_desc[expand_key_1] = word
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
            context_list
    ):
        prompt_list = []
        for context in context_list:
            expand_prompt = self.llm.query(
                user_prompt=self.prompt_expand_prompt
            )
            prompt_list.append(
                expand_prompt
            )
        return prompt_list

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception)
    )
    def overall_expansion(
        self,
        input_context_desc:dict,
        swap_context_desc:dict=None,
        expand_1_key="obj",
        expand_1_type = "synonyms",
    ):
        if swap_context_desc is None:
            context_list = self.content_expansion(
                context_desc=input_context_desc,
                expand_num_1 = 4,
                expand_key_1 = expand_1_key,
                expand_1_type = expand_1_type,
                expand_num_2 = 30
            )
            prompt_list = self.prompt_expansion(context_list=context_list)
            return prompt_list
        else:
            context_list, swap_context_list = self.content_expansion(
                context_desc=input_context_desc,
                expand_num_1=4,
                expand_key_1=expand_1_key,
                expand_1_type=expand_1_type,
                expand_num_2=30,
                swap_value_1=swap_context_desc[expand_1_key]
            )
            real_prompt_list = self.prompt_expansion(context_list=context_list)
            swap_prompt_list = self.prompt_expansion(context_list=swap_context_list)
            return real_prompt_list, swap_prompt_list