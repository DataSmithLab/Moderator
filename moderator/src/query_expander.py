from moderator.src.llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class QueryExpander:
    def __init__(
        self,
        model_name = "llama3.1:8b",
        prompts_folder_path = "moderator/src/prompts",
    ):
        self.model_name = model_name
        self.prompts_folder_path = prompts_folder_path
        self.load_prompt()

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
            
    @retry
    def vocabulary_expand(
        self
    )