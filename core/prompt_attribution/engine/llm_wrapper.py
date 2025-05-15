"""Wrapper for making async API calls to OpenAI models."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ..settings import get_settings


class LLMWrapper:
    """Wrapper for interacting with LLMs via OpenAI's API.
    
    Features:
    - Async batch processing
    - Caching based on prompt hash
    - Automatic retry with exponential backoff
    """
    
    def __init__(self):
        """Initialize the LLM wrapper with settings."""
        settings = get_settings()
        
        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        
        # API settings
        self.completion_model = settings.completion_model
        self.embedding_model = settings.embedding_model
        
        # Performance settings
        self.max_concurrent = settings.max_concurrent_requests
        
        # Cache settings
        self.enable_cache = settings.enable_cache
        self.cache_dir = Path(settings.cache_dir)
        
        # Ensure cache directory exists if caching is enabled
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    async def get_completion(
        self, 
        prompt: str, 
        temperature: float = 0,
        seed: int = 42,
        max_tokens: int = 1000,
    ) -> str:
        """Get a completion from the model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (lower = more deterministic)
            seed: Random seed for reproducibility
            max_tokens: Maximum tokens to generate
            
        Returns:
            The model's completion text
        """
        # Check cache first if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, temperature, seed, max_tokens)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Format the message for the API
        messages = [{"role": "user", "content": prompt}]
        
        # Make the API call with retries
        response = await self._call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.completion_model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
        )
        
        # Extract the completion text
        completion_text = response.choices[0].message.content
        
        # Cache the response if enabled
        if self.enable_cache:
            self._save_to_cache(cache_key, completion_text)
        
        return completion_text
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        # Check cache first if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(text, prefix="embed")
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return json.loads(cached_response)
        
        # Make the API call with retries
        response = await self._call_with_retry(
            lambda: self.client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
        )
        
        embedding = response.data[0].embedding
        
        # Cache the response if enabled
        if self.enable_cache:
            self._save_to_cache(cache_key, json.dumps(embedding))
        
        return embedding
    
    async def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        seed: int = 42,
        max_tokens: int = 1000,
    ) -> str:
        """Get a chat completion from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (lower = more deterministic)
            seed: Random seed for reproducibility
            max_tokens: Maximum tokens to generate
            
        Returns:
            The model's completion text
        """
        # Check cache first if enabled
        if self.enable_cache:
            # Create a string representation of messages for caching
            messages_str = json.dumps(messages)
            cache_key = self._get_cache_key(messages_str, temperature, seed, max_tokens, prefix="chat_msgs")
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Make the API call with retries
        response = await self._call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.completion_model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
        )
        
        # Extract the completion text
        completion_text = response.choices[0].message.content
        
        # Cache the response if enabled
        if self.enable_cache:
            self._save_to_cache(cache_key, completion_text)
        
        return completion_text
    
    async def batch_completions(
        self, 
        prompts: List[str],
        temperature: float = 0,
        seed: int = 42,
        max_tokens: int = 1000,
        show_progress: bool = True,
    ) -> List[str]:
        """Process multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to process
            temperature: Sampling temperature
            seed: Random seed
            max_tokens: Maximum tokens per completion
            show_progress: Whether to show a progress bar
            
        Returns:
            List of completion texts
        """
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_completion(prompt):
            async with semaphore:
                return await self.get_completion(
                    prompt, temperature, seed, max_tokens
                )
        
        for prompt in prompts:
            tasks.append(bounded_completion(prompt))
        
        if show_progress:
            completions = await tqdm_asyncio.gather(*tasks, desc="Processing prompts")
        else:
            completions = await asyncio.gather(*tasks)
            
        return completions
    
    async def _call_with_retry(
        self, 
        api_call, 
        max_retries: int = 3, 
        base_delay: float = 1.0,
        max_delay: float = 16.0,
    ) -> Any:
        """Make API call with exponential backoff retry.
        
        Args:
            api_call: Async function to call
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
        
        Returns:
            API response
            
        Raises:
            Exception: If retries are exhausted
        """
        retries = 0
        delay = base_delay
        
        while True:
            try:
                return await api_call()
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                retries += 1
                if retries > max_retries:
                    raise
                
                # Calculate delay with exponential backoff and jitter
                jitter = 0.1 * delay * (2 * (0.5 - (hash(str(e)) % 10) / 10))
                delay = min(delay * 2 + jitter, max_delay)
                
                # Wait before retrying
                await asyncio.sleep(delay)
    
    def _get_cache_key(self, content: str, *args, prefix: str = "chat") -> str:
        """Generate a cache key from content and args.
        
        Args:
            content: Main content to hash
            *args: Additional args to include in the hash
            prefix: Key prefix for organization
            
        Returns:
            Cache key string
        """
        hash_content = content
        if args:
            hash_content += "||" + "||".join(str(arg) for arg in args)
        
        hash_val = hashlib.sha256(hash_content.encode()).hexdigest()
        return f"{prefix}_{hash_val}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get a response from the cache.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached response if found, None otherwise
        """
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return f.read()
            except Exception:
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, content: str) -> None:
        """Save a response to the cache.
        
        Args:
            cache_key: Cache key to store under
            content: Content to cache
        """
        if not self.enable_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Ensure the cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Write with atomic replacement to avoid partial writes
            temp_file = cache_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                f.write(content)
            temp_file.replace(cache_file)
        except Exception:
            # Silently fail if caching encounters an error
            # This way, the main operation can continue even if caching fails
            pass 