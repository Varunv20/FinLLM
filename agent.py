import os
import anthropic
import openai
from torchao import quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer,  pipeline
import torch
import logging
import math
logger = logging.getLogger(__name__)

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
            "openrouter": llm,
        }[provider]
        

        if provider == "anthropic":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            self.client = openai.OpenAI()
        elif provider == "openrouter":
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost",  # required by OpenRouter (can be your site)
                    "X-Title": "MyAgent"                 # optional but recommended
                }
            )
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

        
    def run(self, prompt: str, max_tokens=500, get_probs=False) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text, {}

        elif self.provider == "openai" or self.provider == "openrouter":
            if get_probs:
                raw_response = self.client.chat.completions.with_raw_response.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    top_logprobs=5,
                )
                logger.info("Response headers: %s", dict(raw_response.headers))
                response = raw_response.parse()

                text = response.choices[0].message.content
                probs = self._extract_decision_probs(response)
                return text, probs

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content, {}

        elif self.provider == "huggingface":
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.client.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = self.client(
                formatted_prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            generated = response[0]["generated_text"]
            return generated[len(formatted_prompt):], {}


    def _extract_decision_probs(self, response, label_map=None):
        """
        Scans the returned logprobs content for the token matching the decision
        (0/1/2 after 'Prediction:') and returns normalized percentages for
        BUY/SELL/HOLD based on that token's top_logprobs alternatives.
        """
        if label_map is None:
            label_map = {"0": "BUY", "1": "SELL", "2": "HOLD"}

        content_tokens = response.choices[0].logprobs.content

        # Find the token that IS one of the digit labels (the model's chosen output token)
        target_index = None
        for i, tok_info in enumerate(content_tokens):
            cleaned = tok_info.token.strip()
            if cleaned in label_map:
                target_index = i
                break

        if target_index is None:
            logger.warning("Could not locate a 0/1/2 token in the output.")
            return None

        top_alts = content_tokens[target_index].top_logprobs

        raw_probs = {name: 0.0 for name in label_map.values()}
        for alt in top_alts:
            cleaned = alt.token.strip()
            if cleaned in label_map:
                name = label_map[cleaned]
                raw_probs[name] = math.exp(alt.logprob)

        total = sum(raw_probs.values())
        if total == 0:
            logger.warning("None of the target digit labels appeared in top_logprobs.")
            return None

        percentages = {k: round((v / total) * 100, 2) for k, v in raw_probs.items()}
        return percentages