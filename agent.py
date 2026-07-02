import os
import anthropic
import openai
from torchao import quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer,  pipeline
import torch

class Agent:
    def __init__(self, provider="huggingface", model=None, llm = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        provider: "anthropic", "openai", or "huggingface"
        model: defaults to a sensible model per provider
        """
        self.provider = provider
        self.model = model or {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "huggingface": llm,
        }[provider]
        

        if provider == "anthropic":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            self.client = openai.OpenAI()
        elif provider == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
            model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,      # or torch.float16
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",       # forces the efficient fused attention path
                )
            if torch.cuda.is_available():
                quantize_(model, Int4WeightOnlyConfig(
                    group_size=32,
                    int4_packing_format="tile_packed_to_4d",
                    int4_choose_qparams_algorithm="hqq",
                ))
            tokenizer = AutoTokenizer.from_pretrained(self.model)

            self.client = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=500, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
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
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.client.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = self.client(
                formatted_prompt,
                max_new_tokens=max_tokens,
                do_sample=False,   # keep deterministic; remove the do_sample=True override
            )
            generated = response[0]["generated_text"]
            return generated[len(formatted_prompt):]