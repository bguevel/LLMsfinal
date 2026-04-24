from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from state import ProofState

TACTICS = ["assumption", "intro", "split", "left", "right", "cases"]


def proof_state_to_text(state: ProofState) -> str:
    if state.is_solved():
        return "Proof is solved."

    g = state.first_goal()
    assert g is not None

    assumptions = ", ".join(str(a) for a in g.assumptions) if g.assumptions else "none"
    goal = str(g.target)

    return (
        f"Assumptions: {assumptions}\n"
        f"Goal: {goal}\n"
        f"Available tactics: {', '.join(TACTICS)}"
    )


class PhiTacticModel:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",)

        self.model.config.use_cache = False
        self.model.eval()

    def build_prompt(self, state: ProofState) -> str:
        return (
            "<|system|>\n"
            "You are a proof-search assistant. "
            "Choose exactly one tactic from the allowed list. "
            "Return only the tactic name.\n"
            "<|end|>\n"
            "<|user|>\n"
            f"{proof_state_to_text(state)}\n\n"
            "Which tactic should be applied next?\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )

    @torch.no_grad()
    def predict_tactic(self, state: ProofState, max_new_tokens: int = 8) -> str:
        prompt = self.build_prompt(state)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generated = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=self.tokenizer.eos_token_id,)

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        response = response.splitlines()[0].strip()
        response = response.split()[0].strip(".,:;")

        return response

    @torch.no_grad()
    def get_state_embedding(self, state: ProofState) -> torch.Tensor:
        prompt = self.build_prompt(state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        vec = last_hidden[:, -1, :]

        return vec.squeeze(0).detach().cpu()