import importlib.util
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

# a model whose bf16 weights take more than this share of GPU memory is quantized on load
# (when it doesn't ship quantized already), leaving room for long transcripts' KV cache
FIT_SHARE = 0.65


class Agent:
    def __init__(self, provider="huggingface", model=None, llm = "meta-llama/Llama-3.2-3B-Instruct",
                 engine=None, max_model_len=32768, quantization="auto", reasoning_effort=None,
                 gpu_memory_utilization=0.90):
        """
        provider: "anthropic", "openai", "openrouter" or "huggingface"
        model: defaults to a sensible model per provider
        engine (huggingface only): "vllm" (batched, much faster; the default when installed)
            or "transformers" (one call at a time)
        max_model_len: longest prompt + reply, in tokens
        quantization: "auto" keeps a model's own quantization (gpt-oss ships MXFP4) and
            otherwise quantizes only if bf16 weights don't fit (fp8 with vllm, int4 with
            transformers); None never quantizes; or a method name ("fp8", "int4")
        reasoning_effort: "low" / "medium" / "high" for models whose chat template supports
            it (gpt-oss); None keeps the model's default
        """
        self.provider = provider
        self.model = model or {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "huggingface": llm,
            "openrouter": llm,
        }[provider]
        self.reasoning_effort = reasoning_effort
        self.max_model_len = max_model_len
        self.tokenizer = None

        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif provider == "openrouter":
            import openai
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost",  # required by OpenRouter (can be your site)
                    "X-Title": "MyAgent"                 # optional but recommended
                }
            )
        elif provider == "huggingface":
            self.engine = engine or ("vllm" if importlib.util.find_spec("vllm") else "transformers")
            from transformers import AutoConfig
            self.config = AutoConfig.from_pretrained(self.model)
            quant = self._pick_quantization(quantization)
            print(f"[agent] {self.model} via {self.engine}; quantization: "
                  f"{self._own_quantization() or quant or 'none (bf16)'}", flush=True)
            if self.engine == "vllm":
                self._load_vllm(quant, gpu_memory_utilization)
            else:
                self._load_transformers(quant)

    # ------------------------------------------------------------------ loading

    def _own_quantization(self):
        """The quantization a checkpoint ships with (e.g. 'mxfp4' for gpt-oss), or None."""
        q = getattr(self.config, "quantization_config", None)
        if q is None:
            return None
        return q.get("quant_method", "quantized") if isinstance(q, dict) else getattr(q, "quant_method", "quantized")

    def _pick_quantization(self, quantization):
        if quantization != "auto":
            return quantization
        if self._own_quantization():
            return None  # never quantize an already-quantized checkpoint again
        try:
            import torch
            from huggingface_hub import get_safetensors_metadata
            n_params = sum(get_safetensors_metadata(self.model).parameter_count.values())
            gpu = torch.cuda.get_device_properties(0).total_memory
        except Exception as e:  # no GPU / metadata: load as is
            logger.warning("could not size %s (%s); loading unquantized", self.model, e)
            return None
        if 2 * n_params <= FIT_SHARE * gpu:
            return None
        return "fp8" if self.engine == "vllm" else "int4"

    def _load_vllm(self, quant, gpu_memory_utilization):
        # FlashInfer's sampler compiles CUDA code on first use, which needs nvcc; cluster GPU
        # nodes often lack it. vLLM's PyTorch sampler needs nothing and decoding is greedy anyway.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        kwargs = dict(model=self.model, max_model_len=self.max_model_len,
                      gpu_memory_utilization=gpu_memory_utilization,
                      # repeated prompt prefixes (task text, emotion story) are computed once
                      enable_prefix_caching=True)
        if quant:
            kwargs["quantization"] = quant
        self.client = LLM(**kwargs)
        self.tokenizer = self.client.get_tokenizer()

    def _load_transformers(self, quant):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self._own_quantization() == "mxfp4" and not importlib.util.find_spec("kernels"):
            print("[agent] WARNING: `kernels` is not installed, so transformers will unpack gpt-oss's "
                  "MXFP4 weights to bf16 (slow, ~4x the memory). pip install 'kernels>=0.17,<0.18'",
                  flush=True)
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True)
        hopper = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
        # fastest attention first; gpt-oss needs attention "sinks", which only the
        # flash-attention-3 kernel and eager support (sdpa raises for it)
        attempts = (["kernels-community/vllm-flash-attn3"] if hopper and importlib.util.find_spec("kernels") else [])
        attempts += ["sdpa", "eager"]
        for attn in attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model, attn_implementation=attn, **load_kwargs)
                break
            except Exception as e:
                if attn == attempts[-1]:
                    raise
                print(f"[agent] attention {attn!r} unavailable ({str(e).splitlines()[0][:120]}); trying the next",
                      flush=True)
        print(f"[agent] attention: {model.config._attn_implementation}", flush=True)
        if quant == "int4" and torch.cuda.is_available():
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
            quantize_(model, Int4WeightOnlyConfig(
                group_size=32,
                int4_packing_format="tile_packed_to_4d",
                int4_choose_qparams_algorithm="hqq",
            ))
        elif quant not in (None, "int4"):
            raise ValueError(f"transformers engine supports quantization None or 'int4', not {quant!r}")
        model.eval()
        self.client = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    # ------------------------------------------------------------------ prompts

    def format(self, prompt):
        """The prompt wrapped in the model's chat template, as token ids."""
        kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, **kwargs)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def truncate(self, text, max_tokens):
        """`text` cut to its first `max_tokens` tokens, and whether it was cut."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text, False
        return self.tokenizer.decode(ids[:max_tokens]), True

    # ------------------------------------------------------------------ generation

    def run_batch(self, prompts, max_tokens=2048, force=None):
        """Replies to many prompts. Each is a dict: text, input_tokens, output_tokens,
        hit_max_tokens (the reply was cut off at max_tokens) and forced. vllm runs them
        concurrently.

        force: {"prefix": "Decision:", "choices": ["0", "1", "2"], "done": text -> bool}. A reply
        cut off before done(text) is true gets an answer forced: its reasoning is closed, the
        prefix written, and the model picks the most likely of the choices (budget forcing)."""
        results, gen_ids = self._generate(prompts, max_tokens)
        for r in results:
            r["forced"] = False
        if force:
            todo = [k for k, r in enumerate(results) if r["hit_max_tokens"] and not force["done"](r["text"])]
            if todo:
                self._force(prompts, results, gen_ids, todo, force)
        return results

    def _generate(self, prompts, max_tokens):
        ids = [self.format(p) for p in prompts]
        too_long = [len(i) + max_tokens > self.max_model_len for i in ids]
        if any(too_long):
            raise ValueError(f"{sum(too_long)} prompts + max_tokens exceed max_model_len={self.max_model_len}; "
                             "lower max_input_tokens or raise max_model_len")
        start = time.time()
        if self.engine == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=True)
            outs = self.client.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False)
            results = [dict(text=o.outputs[0].text, input_tokens=len(i), output_tokens=len(o.outputs[0].token_ids),
                            hit_max_tokens=o.outputs[0].finish_reason == "length") for o, i in zip(outs, ids)]
            gen_ids = [list(o.outputs[0].token_ids) for o in outs]
        else:
            import torch
            results, gen_ids = [], []
            for i in ids:
                x = torch.tensor([i], device=self.client.device)
                with torch.inference_mode():
                    out = self.client.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_tokens,
                                               do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                new = out[0, len(i):]
                results.append(dict(text=self.tokenizer.decode(new, skip_special_tokens=True), input_tokens=len(i),
                                    output_tokens=len(new), hit_max_tokens=len(new) >= max_tokens))
                gen_ids.append(new.tolist())
        elapsed = time.time() - start
        n_out = sum(r["output_tokens"] for r in results)
        print(f"[agent] {len(prompts)} prompts, {sum(r['input_tokens'] for r in results)} in / {n_out} out "
              f"tokens in {elapsed:.0f}s ({n_out / max(elapsed, 1e-9):.0f} output tok/s)", flush=True)
        return results, gen_ids

    def _close_reasoning(self, text):
        """(tokens that end an unfinished reasoning block, how they read in decoded text), or
        None if the reply already got past its reasoning."""
        if "gpt-oss" in self.model.lower():
            # harmony format: analysis channel, then a final-channel message
            if "assistantfinal" in text:
                return None
            return "<|end|><|start|>assistant<|channel|>final<|message|>", "assistantfinal"
        if "<think>" in text and "</think>" not in text:
            return "\n</think>\n\n", "\n</think>\n\n"
        return None

    def _force(self, prompts, results, gen_ids, todo, force):
        """Budget forcing: append the reasoning close and prefix to each cut-off reply and take
        the most likely allowed choice as the next token."""
        enc = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        # the choices as single tokens after the prefix (" 0" usually is one token; else "0")
        choice_ids = [enc(" " + c) for c in force["choices"]]
        sep = " "
        if any(len(c) != 1 for c in choice_ids):
            choice_ids, sep = [enc(c) for c in force["choices"]], ""
        if any(len(c) != 1 for c in choice_ids):
            raise ValueError(f"choices {force['choices']} aren't single tokens for {self.model}")
        choice_ids = [c[0] for c in choice_ids]
        prefix = force["prefix"] + ("" if sep else " ")
        conts, shown = [], []
        for k in todo:
            close = self._close_reasoning(results[k]["text"])
            tail, read = close if close else ("\n", "\n")
            conts.append(self.format(prompts[k]) + gen_ids[k] + enc(tail + prefix))
            shown.append(read + prefix + sep)
        too_long = sum(len(c) + 1 > self.max_model_len for c in conts)
        if too_long:
            raise ValueError(f"{too_long} forced continuations exceed max_model_len={self.max_model_len}")
        if self.engine == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(temperature=0.0, max_tokens=1, allowed_token_ids=choice_ids)
            outs = self.client.generate([{"prompt_token_ids": c} for c in conts], params, use_tqdm=False)
            picks = [o.outputs[0].token_ids[0] for o in outs]
        else:
            import torch
            picks = []
            for c in conts:
                x = torch.tensor([c], device=self.client.device)
                with torch.inference_mode():
                    logits = self.client(x).logits[0, -1]
                picks.append(choice_ids[int(torch.argmax(logits[choice_ids]))])
        label = dict(zip(choice_ids, force["choices"]))
        for k, s, p in zip(todo, shown, picks):
            results[k]["text"] += s + label[p]
            results[k]["forced"] = True
        print(f"[agent] forced a decision on {len(todo)} cut-off replies", flush=True)

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
            return self.run_batch([prompt], max_tokens=max_tokens)[0]["text"], {}


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
