"""Emotion by activation steering: the neutral prompt for every call, with an emotion vector
added to the model's residual stream instead of an emotion story in the prompt.

    python steer_main.py --summaries                     # neutral + the 11 emotions, all calls
    python steer_main.py --summaries --smoke             # 4 calls, neutral / anger / excitement
    python steer_main.py --summaries --strength 0.2 --layer 14

Vectors: each emotion story from prompts.py is run through the model and its residual-stream
activations after `--layer` are averaged over the story's tokens; an emotion's vector is its
story's mean minus the mean over all 11 stories (so what the stories share, "a story about an
analyst's morning", cancels), scaled to unit length. While a call runs, strength x (the
layer's typical activation norm) x vector is added to the output of that layer's attention
projection (o_proj), at every position, prompt and reply. "neutral" runs with no vector, so
it is the control for this experiment.

Engines: with vllm (the default when installed) the vector goes into o_proj's bias, edited in
place between emotions, so runs are as fast as main.py; this needs a model whose attention
has a bias (gpt-oss does). The vectors themselves are computed once with transformers and
cached. --engine transformers adds the same vector with a hook instead (any model, slower).

Results: steer/s<strength>_L<layer>/<prefix><model>_<neutral|0..10>.pkl (same columns and
resume behaviour as main.py), and the vectors in the same folder as vectors.pt. Calls are
main.py's call list, so results line up call by call with the prompt-story runs:

    python emotion_compare.py "steer/s0.10_L12/anon_sum_openai/gpt-oss-20b_*.pkl" --out results_steer/emotions
"""
import argparse
import functools
import gc
import importlib.util
import os
import re

from main import run_all, select_calls
from llms import load_catalogue, MAX_INPUT_TOKENS, MAX_NEW_TOKENS
from agent import Agent
from prompts import emotion_stories

import torch

# LLM.apply_model ships a function to vLLM's engine process, which needs pickling allowed
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _o_proj(model, layer):
    """The attention output projection of decoder layer `layer` (transformers or vllm naming)."""
    pat = re.compile(rf"layers\.{layer}\.(self_attn|attn)\.o_proj$")
    found = [m for n, m in model.named_modules() if pat.search(n)]
    if len(found) != 1:
        raise ValueError(f"found {len(found)} o_proj modules for layer {layer}")
    return found[0]


def _set_bias(model, layer, vec):
    """vllm side: o_proj bias of `layer` = its original value + vec (None restores it)."""
    m = _o_proj(model, layer)
    if m.bias is None:
        raise ValueError("this model's attention has no bias to steer through; use --engine transformers")
    if not hasattr(m, "_steer_orig"):
        m._steer_orig = m.bias.data.clone()
    m.bias.data.copy_(m._steer_orig if vec is None else m._steer_orig + vec.to(m.bias.device, m.bias.dtype))
    return True


class SteeredAgent(Agent):
    """An Agent that adds a vector to one layer's attention output while it generates."""

    def __init__(self, layer, engine=None, **kwargs):
        engine = engine or ("vllm" if importlib.util.find_spec("vllm") else "transformers")
        super().__init__(provider="huggingface", engine=engine, **kwargs)
        self.layer = layer
        self.vector = None
        if self.engine == "transformers":
            _o_proj(self.client, layer).register_forward_hook(self._hook)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[steer] steering layer {layer} via {self.engine}", flush=True)

    def set_vector(self, vec):
        """The vector added from now on (already scaled), or None for no steering."""
        self.vector = vec
        if self.engine == "vllm":
            self.client.apply_model(functools.partial(_set_bias, layer=self.layer,
                                                      vec=None if vec is None else vec.cpu()))
            # cached prompt prefixes were computed under the previous vector
            self.client.reset_prefix_cache()

    def _hook(self, module, inputs, output):
        if self.vector is None:
            return None
        return output + self.vector.to(output.device, output.dtype)

    # ------------------------------------------------------------------ vectors

    def emotion_vectors(self, stories):
        """{emotion: unit vector} from each story's mean activation after self.layer (over
        its own tokens, not the chat template around it), and the mean per-token norm there.
        transformers engine only."""
        texts = list(stories.values())
        ids = [self.format(t) for t in texts]
        # the template is the same around every text: strip its common prefix and suffix
        pre = 0
        while all(len(i) > pre and i[pre] == ids[0][pre] for i in ids):
            pre += 1
        suf = 0
        while all(len(i) > pre + suf and i[-1 - suf] == ids[0][-1 - suf] for i in ids):
            suf += 1
        means, norms = [], []
        prev, self.vector = self.vector, None
        try:
            for i in ids:
                x = torch.tensor([i], device=self.client.device)
                with torch.inference_mode():
                    h = self.client(x, output_hidden_states=True).hidden_states[self.layer + 1][0]
                h = h[pre:len(i) - suf].float()
                means.append(h.mean(0))
                norms.append(h.norm(dim=-1).mean())
        finally:
            self.vector = prev
        acts = torch.stack(means)
        diffs = acts - acts.mean(0)
        unit = diffs / diffs.norm(dim=-1, keepdim=True)
        return dict(zip(stories, unit.cpu())), float(torch.stack(norms).mean())

    # ------------------------------------------------------------------ generation

    def _generate(self, prompts, max_tokens):
        """vllm as in Agent; transformers batched and left-padded (Agent's runs one at a time)."""
        if self.engine == "vllm":
            return super()._generate(prompts, max_tokens)
        import time
        ids = [self.format(p) for p in prompts]
        too_long = [len(i) + max_tokens > self.max_model_len for i in ids]
        if any(too_long):
            raise ValueError(f"{sum(too_long)} prompts + max_tokens exceed max_model_len={self.max_model_len}")
        start = time.time()
        width = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        x = torch.tensor([[pad] * (width - len(i)) + i for i in ids], device=self.client.device)
        mask = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids], device=self.client.device)
        with torch.inference_mode():
            out = self.client.generate(x, attention_mask=mask, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=pad)
        eos = self.client.generation_config.eos_token_id
        stops = {pad, self.tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])} - {None}
        results, gen_ids = [], []
        for i, row in zip(ids, out[:, width:].tolist()):
            n = next((k + 1 for k, t in enumerate(row) if t in stops), len(row))
            new = row[:n]
            results.append(dict(text=self.tokenizer.decode(new, skip_special_tokens=True), input_tokens=len(i),
                                output_tokens=len(new), hit_max_tokens=n >= max_tokens and new[-1] not in stops))
            gen_ids.append([t for t in new if t not in stops])
        elapsed = time.time() - start
        n_out = sum(r["output_tokens"] for r in results)
        print(f"[steer] {len(prompts)} prompts, {sum(r['input_tokens'] for r in results)} in / {n_out} out "
              f"tokens in {elapsed:.0f}s ({n_out / max(elapsed, 1e-9):.0f} output tok/s)", flush=True)
        return results, gen_ids


def load_vectors(path, agent_kwargs, layer, agent=None):
    """The cached vectors at `path`, else computed (on `agent` if it runs transformers, else
    on a transformers copy of the model loaded just for this and freed after) and saved."""
    if os.path.exists(path):
        saved = torch.load(path)
        if saved["layer"] == layer:
            return saved["vectors"], saved["norm"]
    tmp = None
    if agent is None or agent.engine != "transformers":
        tmp = agent = SteeredAgent(layer=layer, engine="transformers", **agent_kwargs)
    vectors, norm = agent.emotion_vectors(emotion_stories)
    if tmp is not None:
        del tmp, agent
        gc.collect()
        torch.cuda.empty_cache()
    torch.save({"layer": layer, "norm": norm, "vectors": vectors}, path)
    return vectors, norm


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--engine", choices=["vllm", "transformers"], help="default: vllm if installed")
    ap.add_argument("--layer", type=int, help="layer to steer (default: the middle one)")
    ap.add_argument("--strength", type=float, default=0.1,
                    help="vector length as a share of the layer's typical activation norm")
    ap.add_argument("--year", type=int, default=2018, help="calls from this year (0 = all years)")
    ap.add_argument("--n", type=int, help="sample this many usable calls (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calls-file", help="fixed call list (default: main.py's, runs/<prefix><model>_<year>_calls.csv)")
    ap.add_argument("--original", action="store_true", help="read the original, not anonymized, transcripts")
    ap.add_argument("--summaries", action="store_true",
                    help="read each call's summary (summarize_calls.py) instead of the full transcript")
    ap.add_argument("--prompts", default="all", help="'all', 'neutral', or comma-separated emotions")
    ap.add_argument("--batch-size", type=int, help="default: 64 with vllm, 32 with transformers")
    ap.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--quantization", default="auto", help="auto / none / fp8 / int4")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="gpt-oss only; default: the model's own (medium)")
    ap.add_argument("--retry-cut", action="store_true",
                    help="rerun calls whose saved reply hit the token limit (use with a larger --max-new-tokens)")
    ap.add_argument("--no-force-decision", action="store_true",
                    help="leave replies cut off before their decision without one (default: force it)")
    ap.add_argument("--smoke", action="store_true",
                    help="4 calls; neutral, anger and excitement; results in runs/smoke/steer/")
    return ap.parse_args()


if __name__ == "__main__":
    from transformers import AutoConfig
    args = parse_args()
    catalogue = load_catalogue()
    if args.year:
        catalogue = catalogue[catalogue["call_date"].dt.year == args.year]
    anonymized = not args.original
    base = "anon_" if anonymized else "sample_"
    prefix = base + ("sum_" if args.summaries else "")
    model_tag = args.model.replace("/", "_")
    calls_file = args.calls_file or f"runs/{base}{model_tag}_{args.year or 'all'}_calls.csv"
    calls = select_calls(catalogue, calls_file, n=args.n, seed=args.seed)
    if args.summaries:
        from llms import load_summaries
        sums = load_summaries(columns=["call_id", "summary"])
        calls = calls[calls["call_id"].isin(set(sums.loc[sums["summary"].str.len() > 0, "call_id"]))]
        if calls.empty:
            raise SystemExit("no summaries to run on: python summarize_calls.py first")
    emotions = list(emotion_stories)
    if args.prompts == "all":
        prompts = [None] + emotions
    else:
        prompts = [None if p == "neutral" else p for p in args.prompts.split(",")]
    if args.smoke:
        calls, prompts = calls.head(4), [None, "anger", "excitement"]

    config = AutoConfig.from_pretrained(args.model)
    n_layers = getattr(config, "num_hidden_layers", None) or config.get_text_config().num_hidden_layers
    layer = n_layers // 2 if args.layer is None else args.layer
    folder = ("runs/smoke/steer/" if args.smoke else "steer/") + f"s{args.strength:.2f}_L{layer}/"
    os.makedirs(folder, exist_ok=True)
    agent_kwargs = dict(llm=args.model, max_model_len=args.max_model_len,
                        quantization=None if args.quantization == "none" else args.quantization,
                        reasoning_effort=args.reasoning_effort)

    engine = args.engine or ("vllm" if importlib.util.find_spec("vllm") else "transformers")
    agent = None
    if engine == "transformers":  # one model for both the vectors and the run
        agent = SteeredAgent(layer=layer, engine=engine, **agent_kwargs)
    vectors, norm = load_vectors(folder + "vectors.pt", agent_kwargs, layer, agent)
    if agent is None:
        agent = SteeredAgent(layer=layer, engine=engine, **agent_kwargs)
    cos = torch.stack(list(vectors.values()))
    print(f"[steer] layer {layer} of {n_layers}: typical norm {norm:.1f}, added vector length "
          f"{args.strength * norm:.1f}; mean |cosine| between emotions "
          f"{(cos @ cos.T).abs().fill_diagonal_(0).sum() / (len(cos) ** 2 - len(cos)):.2f}", flush=True)
    print(f"{len(calls)} calls x {len(prompts)} prompts ({calls_file}) -> {folder}", flush=True)

    batch_size = args.batch_size or (64 if agent.engine == "vllm" else 32)
    for e in prompts:
        label = "neutral" if e is None else str(emotions.index(e))
        agent.set_vector(None if e is None else args.strength * norm * vectors[e])
        result = run_all(agent, calls, name=folder + prefix + args.model + "_" + label, prompt=None,
                         anonymized=anonymized, batch_size=batch_size,
                         max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                         summaries=args.summaries, retry_cut=args.retry_cut,
                         force_decision=not args.no_force_decision)
        if args.smoke and len(result):
            print(f"\n=== {e or 'neutral'} ===")
            print(result[["ticker", "decision", "output_tokens", "hit_max_tokens", "forced"]].to_string())
            print("last reply:\n" + str(result["summary"].iloc[-1])[-800:])
