
"""
AutoGen 0.6.1 模型客户端适配器
"""

import os
from typing import Optional, Dict, Any

class ModelClientAdapter:
    """模型客户端适配器，支持多种AI模型"""
    
    def __init__(self):
        self.clients = {}
    
    def create_openai_client(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """创建OpenAI客户端"""
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            client = OpenAIChatCompletionClient(
                model=model,
                api_key=api_key
            )
            self.clients['openai'] = client
            return client
            
        except ImportError as e:
            raise ImportError(f"Failed to import OpenAI client: {e}")
    
    def create_anthropic_client(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        """创建Anthropic客户端"""
        try:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient
            
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            
            client = AnthropicChatCompletionClient(
                model=model,
                api_key=api_key
            )
            self.clients['anthropic'] = client
            return client
            
        except ImportError:
            # 如果Anthropic客户端不可用，使用OpenAI作为后备
            print("Warning: Anthropic client not available, using OpenAI as fallback")
            return self.create_openai_client()
    
    def create_google_client(self, model: str = "gemini-pro", api_key: Optional[str] = None):
        """创建Google客户端"""
        try:
            from autogen_ext.models.google import GoogleChatCompletionClient
            
            api_key = api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not provided")
            
            client = GoogleChatCompletionClient(
                model=model,
                api_key=api_key
            )
            self.clients['google'] = client
            return client
            
        except ImportError:
            # 如果Google客户端不可用，使用OpenAI作为后备
            print("Warning: Google client not available, using OpenAI as fallback")
            return self.create_openai_client()
    
    def get_client(self, provider: str):
        """获取指定提供商的客户端"""
        return self.clients.get(provider)
