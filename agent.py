import os
import anthropic
import openai
from transformers import pipeline

class Agent:
    def __init__(self, provider="huggingface", model=None):
        """
        provider: "anthropic", "openai", or "huggingface"
        model: defaults to a sensible model per provider
        """
        self.provider = provider
        self.model = model or {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
        }[provider]

        if provider == "anthropic":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            self.client = openai.OpenAI()
        elif provider == "huggingface":
            from transformers import pipeline, BitsAndBytesConfig
            import torch

            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            self.client = pipeline(
                "text-generation",
                model=self.model,
                device_map="auto",
                model_kwargs={"quantization_config": quant_config},
                torch_dtype=torch.float16,
                max_new_tokens=500,
                do_sample=False,        # deterministic, better for predictions
                pad_token_id=2,
            )

    def run(self, prompt: str, max_tokens=500) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        elif self.provider == "huggingface":
            response = self.client(prompt, max_new_tokens=max_tokens, do_sample=True)
            return response[0]["generated_text"][len(prompt):]