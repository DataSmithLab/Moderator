import json
from typing import Optional

class ContextDesc:
    def __init__(
            self,
            obj:str=None,
            sty:str=None,
            act:str=None
        ) -> None:
        self.obj = obj
        self.sty = sty
        self.act = act
    
    def to_dict(self):
        return {
            "obj":self.obj,
            "sty":self.sty,
            "act":self.act
        }
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict())
    
    def get_blank_contexts(self):
        blank_contexts = []
        for key, value in self.to_dict().items():
            if value is None or value == "":
                blank_contexts.append(key)
        return blank_contexts
    
    def get_context(self, context_key:str):
        return self.to_dict()[context_key]
    
    def set_context(self, context_key:str, context_value:str):
        if context_key == "obj":
            self.obj = context_value
        elif context_key == "sty":
            self.sty = context_value
        elif context_key == "act":
            self.act = context_value
        else:
            raise ValueError("context_key must be obj, sty or act")