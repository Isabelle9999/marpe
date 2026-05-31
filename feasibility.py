# feasibility check: two llama-3.2-3b at bf16 + one grpo-like backward on rtx 4080

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

try:
    import flash_attn  # noqa: F401
    ATTN = "flash_attention_2"
except ImportError:
    ATTN = "sdpa"

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEV = "cuda:0"
BUDGET_GB = 14.0
ART = Path("./artifacts/feasibility")
PROMPT = "hello, how are you today?"


def main():
    ART.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "no cuda device"
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("loading agent a (bf16 + lora)")
    a = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=ATTN
    )
    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    a = get_peft_model(a, lora)
    a.gradient_checkpointing_enable()
    a.enable_input_require_grads()
    a.config.use_cache = False

    print("loading agent b (bf16, frozen)")
    b = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=ATTN
    )
    b.eval()
    for p in b.parameters():
        p.requires_grad = False

    load_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak vram after load: {load_peak:.2f} gb")

    torch.cuda.reset_peak_memory_stats()
    ids = tok(PROMPT, return_tensors="pt").to(DEV)
    with torch.no_grad():
        a.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        b.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
    gen_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak vram after 4-token gen from each: {gen_peak:.2f} gb")

    torch.cuda.reset_peak_memory_stats()
    # grpo-like: 4 rollouts from a w/o grad, then re-forward with grad for log-probs
    enc = tok([PROMPT] * 1, return_tensors="pt", padding=True).to(DEV)
    torch.manual_seed(0)
    with torch.no_grad():
        rollouts = a.generate(
            **enc,
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
    attn = (rollouts != tok.pad_token_id).long()
    with torch.no_grad(), a.disable_adapter():
        ref_logits = a(rollouts, attention_mask=attn).logits
    logits = a(rollouts, attention_mask=attn).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = rollouts[:, 1:].unsqueeze(-1)
    chosen = logp.gather(-1, tgt).squeeze(-1)
    loss = -(chosen * attn[:, 1:]).sum() / attn[:, 1:].sum()
    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak vram after backward: {bwd_peak:.2f} gb")

    deviations = [
        "model: Llama-3.2-3B-Instruct in BF16 (fallback from Meta-Llama-3-8B-Instruct 8-bit x 2, which OOMed during gradient forward on 16GB; documented feasibility check fail action per CLAUDE.md)",
    ]
    deviations.append("rollout config: batch 1 x seq 256 per-step (matches Obfuscation Atlas max_new_tokens=256; effective batch 8 will be recovered via gradient_accumulation_steps=8 in RL training per standard consumer-hardware GRPO practice)")
    if ATTN == "sdpa":
        deviations.append("attn_implementation: sdpa (flash_attn unavailable; deviation from Obfuscation Atlas flash_attention_2 default)")
    (ART / "vram_log.txt").write_text(
        f"load_peak_gb {load_peak:.4f}\n"
        f"gen_peak_gb {gen_peak:.4f}\n"
        f"bwd_peak_gb {bwd_peak:.4f}\n"
        + "".join(f"deviation {d}\n" for d in deviations)
    )

    worst = max(load_peak, gen_peak, bwd_peak)
    if worst < BUDGET_GB:
        print(f"pass: worst peak {worst:.2f} gb < {BUDGET_GB:.1f} gb budget")
    else:
        print(f"fail: worst peak {worst:.2f} gb >= {BUDGET_GB:.1f} gb budget")
        print("fallback: meta-llama/Llama-3.2-3B-Instruct in bf16")


if __name__ == "__main__":
    main()
