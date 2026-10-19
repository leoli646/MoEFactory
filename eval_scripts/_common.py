"""通用工具：加载 base model + moe_lora checkpoint，给生成式评估脚本复用。"""

import argparse
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 把仓库 src/ 加到 PYTHONPATH，以便 import llamafactory
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from llamafactory.model.model_utils.moe_lora import load_moe_lora_state  # noqa: E402
from llamafactory.model.model_utils.das_lora import load_das_lora_state  # noqa: E402
from llamafactory.model.model_utils.weight_peft import load_weight_peft_state  # noqa: E402
from llamafactory.model.model_utils.partial_ft import load_partial_ft_state  # noqa: E402


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--base_model", required=True, help="HF base model id, e.g. allenai/OLMoE-1B-7B-0924")
    p.add_argument("--adapter_path", default=None, help="moe_lora checkpoint dir; omit to evaluate base model only")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="只评估前 N 个样本（debug 用）")
    p.add_argument("--batch_size", type=int, default=1, help="生成 batch size")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--save_path", default=None, help="把每个样本的 prediction 保存到 jsonl")
    p.add_argument(
        "--trust_remote_code",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="是否允许自定义远程代码（默认 true）。DeepSeek-V2-Lite 自带的老 modeling_deepseek.py "
             "与 transformers 5.6 不兼容（缺 is_torch_fx_available），传 false 走原生 deepseek_v2。",
    )
    return p


def load(args) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # decoder-only 模型 batch generate 必须 left-padding，否则模型会从 PAD 之后续写，输出全废
    tokenizer.padding_side = "left"

    # 多卡 (torchrun)：每个 rank 加载一份模型到自己的 GPU，避免 device_map="auto"
    # 把 shared moe_lora 模块和不同 layer 分到不同卡上引发跨设备调用。
    # 单卡：用 device_map="auto" 让 transformers 自己处理。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model.to(f"cuda:{local_rank}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
    if args.adapter_path:
        # 自动识别 adapter 类型:
        #   有 das_lora_state.safetensors    → DAS-LoRA(自家实现)
        #   有 moe_lora_state.safetensors    → MoE-LoRA(自家实现)
        #   有 adapter_config.json           → 标准 LoRA(PEFT)
        das_state = os.path.join(args.adapter_path, "das_lora_state.safetensors")
        moe_state = os.path.join(args.adapter_path, "moe_lora_state.safetensors")
        weight_peft_state = os.path.join(args.adapter_path, "weight_peft_state.safetensors")
        partial_ft_state = os.path.join(args.adapter_path, "partial_ft_state.safetensors")
        peft_config = os.path.join(args.adapter_path, "adapter_config.json")
        if os.path.exists(weight_peft_state):
            print(f"Loading weight_peft adapter from {args.adapter_path}")
            model = load_weight_peft_state(model, args.adapter_path)
        elif os.path.exists(partial_ft_state):
            print(f"Loading partial_ft checkpoint from {args.adapter_path}")
            model = load_partial_ft_state(model, args.adapter_path)
        elif os.path.exists(das_state):
            print(f"Loading DAS-LoRA adapter from {args.adapter_path}")
            model = load_das_lora_state(model, args.adapter_path)
        elif os.path.exists(moe_state):
            print(f"Loading MoE-LoRA adapter from {args.adapter_path}")
            model = load_moe_lora_state(model, args.adapter_path)
        elif os.path.exists(peft_config):
            print(f"Loading PEFT (standard) LoRA adapter from {args.adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=dtype)
            model = model.merge_and_unload()  # 合并 LoRA 到 base 权重,推理更快
        else:
            raise FileNotFoundError(
                f"{args.adapter_path} 既无 das_lora_state.safetensors / "
                f"moe_lora_state.safetensors 也无 adapter_config.json,无法识别 adapter 类型"
            )
    else:
        print("No --adapter_path provided, evaluating BASE model")
    model.eval()
    return tokenizer, model


def _collect_eos_ids(model, tokenizer):
    """收集所有应当终止生成的 token id。

    Qwen3-Base 的 generation_config.eos_token_id=151643(<|endoftext|>)不含 chatml 的
    <|im_end|>(151645)；若不显式传给 generate,模型生成 <|im_end|> 后不会停,而是继续
    伪造下一轮 user 对话(复述题干),污染答案解析。这里取 tokenizer eos +
    generation_config eos + <|im_end|>(若在词表中)的并集。
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc_eos is not None:
        ids.update(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(ids)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature=0.0, top_p=0.95):
    """Batch 生成。返回纯 completion（不含 prompt）。

    temperature=0.0 时用 greedy；>0 时用 sampling。
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=_collect_eos_ids(model, tokenizer),
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**gen_kwargs)
    # left-padding 下,真正的生成部分从 padded 总长之后开始;
    # 用"非 pad token 数"当起点会把 prompt 尾部泄露到 completion 里。
    input_len = enc.input_ids.shape[1]
    completions = []
    for ids in out:
        completions.append(tokenizer.decode(ids[input_len:], skip_special_tokens=True))
    return completions


def wrap_default_chat(tokenizer, user_content: str) -> str:
    """对齐 LlamaFactory `template: default` 训练格式:
        Human: {content}<EOS>
        Assistant:
    生成端不带 trailing EOS,模型从 'Assistant:' 后开始续写。
    """
    eos = tokenizer.eos_token or "<|endoftext|>"
    return f"Human: {user_content}{eos}\nAssistant:"


def wrap_for_model(tokenizer, user_content: str, base_model: str, enable_thinking: bool = False,
                   nonqwen_wrap: str = "plain") -> str:
    """根据 base model 自动选择 chat template 包装方式。

    - Qwen3 (base_model 含 'qwen' 且 tokenizer 有 chat_template):
      用 Qwen3 chat template。enable_thinking 默认 False(避免 <think> 占满
      max_new_tokens,保持既有评测数字不变);需要思考的评测
      显式传 True,模板不注入空 <think></think>,由模型自主决定是否思考。
    - 其他 (gemma4 / gpt-oss / OLMoE / DeepSeek 等) 按 `nonqwen_wrap`:
      * "plain"(默认): 原样返回 plain prompt,保持向后兼容,不影响既有数字。
      * "default": 套 LlamaFactory `template: default` 的 `Human:/Assistant:`
        包装(见 wrap_default_chat)。这些模型的 math 训练用的正是 template=default
        (qwen3 用 template=qwen3 走上面的分支),推理端套 default 才与训练分布一致。
        评测端喂裸 prompt 会造成训练/推理 template 不对齐 → 分数受损。

    注意:非 qwen 模型**不该**套它们各自的原生 chat template(gpt-oss harmony /
    gemma4 turn):训练没用它,且 gpt-oss harmony 会激活 analysis/reasoning channel。
    要对齐的是训练用的 default 模板,不是模型原生模板。

    LlamaFactory 训练时 `template: qwen3` 与 tokenizer 自带 chat_template 对齐,
    所以 qwen3 推理端 apply_chat_template 与训练分布一致。
    """
    is_qwen3 = (
        tokenizer.chat_template is not None
        and "qwen" in str(base_model).lower()
    )
    if is_qwen3:
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # 老版 transformers 不支持 enable_thinking 参数
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    if nonqwen_wrap == "default":
        return wrap_default_chat(tokenizer, user_content)
    return user_content




