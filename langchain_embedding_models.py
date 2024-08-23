import os
from dotenv import load_dotenv
from typing import Optional
from langchain_openai.embeddings import OpenAIEmbeddings


class MyOpenAIEmbeddings:
    @classmethod
    def from_model(
        cls, 
        model: str = 'small',
        *,
        dimensions: Optional[int] = None,
        max_retries: int = 1,
        **kwargs
        )-> OpenAIEmbeddings:
        
        if model in ['text-embedding-3-small', 'TEXT-EMBEDDING-3-SMALL', 'small', 'SMALL']:
            model = 'text-embedding-3-small'
            dimensions = 1536 if dimensions is None else dimensions
        elif model in ['text-embedding-3-large', 'TEXT-EMBEDDING-3-LARGE', 'large', 'LARGE']:
            model = 'text-embedding-3-large'
            dimensions = 3072 if dimensions is None else dimensions
        else:
            raise ValueError(f"Model {model} is currently not supported. Supported models are: ['text-embedding-3-small', 'text-embedding-3-large']")

        load_dotenv()
        return OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), 
            model=model,
            dimensions=dimensions,
            max_retries=max_retries,
            **kwargs
            )


    @classmethod
    def get_model_price(cls)-> dict:
        # Dictionary to store the cost of input and output tokens for each model
        supported_models = {'text-embedding-3-small' : 0.02}  # text-embedding-3-small model: $0.02 per 1M tokens
        supported_models.update({'text-embedding-3-large' : 0.13})  # text-embedding-3-large model: $0.13 per 1M tokens

        return supported_models



