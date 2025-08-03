from moderator.src.context_desc import ContextDesc

class ContentConfig:
    def __init__(
            self,
            content_name:str,
            label_content:ContextDesc,
            real_content:ContextDesc
        ) -> None:
        self.content_name = content_name
        self.label_content = label_content
        self.real_content = real_content
    
    def to_dict(self):
        return {
            "content_name": self.content_name,
            "label_content": self.label_content.to_dict(),
            "real_content": self.real_content.to_dict()
        }
    
    def __str__(self) -> str:
        return '''
        ContentConfig(
            content_name={content_name},
            label_content={label_content},
            real_content={real_content}
        )'''.format(
            content_name=self.content_name,
            label_content=self.label_content,
            real_content=self.real_content
        )