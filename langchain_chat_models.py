import os
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI


class MyChatOpenAI:
    @classmethod
    def from_model(
        cls, 
        model: str = 'gpt-4o-mini',
        *,
        langsmith_project: str = 'default',
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        max_retries: int = 1,
        **kwargs
        )-> ChatOpenAI:
        
        os.environ['LANGCHAIN_PROJECT'] = langsmith_project
        if model in ['gpt-4o', 'GPT-4o', 'GPT-4O', 'gpt-4O', 'gpt4o', 'GPT4o', 'GPT4O', 'gpt4O']:
            model = 'gpt-4o'
        elif model in ['gpt-4o-mini', 'GPT-4o-mini', 'GPT-4O-mini', 'gpt-4O-mini', 'gpt4o-mini', 'GPT4o-mini', 'GPT4O-mini', 'gpt4O-mini', 'gpt4omini', 'GPT4omini', 'GPT4Omini', 'gpt4Omini']:
            model = 'gpt-4o-mini'
        else:
            raise ValueError(f"Model {model} is currently not supported. Supported models are: ['gpt-4o', 'gpt-4o-mini']")

        load_dotenv()
        return ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), 
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs
            )


    @classmethod
    def get_model_price(cls)-> dict:
        # Dictionary to store the cost of input and output tokens for each model
        supported_models = {'gpt-4o' : (5, 15)}  # gpt-4o model: input cost = $5 per 1M tokens, output cost = $15 per 1M tokens
        supported_models.update({'gpt-4o-mini' : (0.15, 0.6)})  # gpt-4o-mini model: input cost = $0.15 per 1M tokens, output cost = $0.6 per 1M tokens

        return supported_models



