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


class QwenTacticModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def build_messages(self, state: ProofState):
        prompt = (
            "You are helping with a propositional-logic proof assistant.\n"
            "Choose exactly one next tactic from this list:\n"
            f"{', '.join(TACTICS)}\n\n"
            "Return only the tactic name and nothing else.\n\n"
            f"{proof_state_to_text(state)}\n"
            "Next tactic:"
        )

        return [
            {
                "role": "system",
                "content": "You are a careful reasoning assistant that outputs only one tactic name."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    @torch.no_grad()
    def predict_tactic(self, state: ProofState, max_new_tokens: int = 8) -> str:
        messages = self.build_messages(state)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Keep only first line/token-ish chunk
        response = response.splitlines()[0].strip()
        response = response.split()[0].strip(".,:;")

        return response

    @torch.no_grad()
    def get_state_embedding(self, state: ProofState) -> torch.Tensor:
        """
        Returns a pooled hidden-state vector for probing later.
        """
        messages = self.build_messages(state)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # last layer, last token
        last_hidden = outputs.hidden_states[-1]   # [batch, seq, hidden]
        vec = last_hidden[:, -1, :]               # [batch, hidden]
        return vec.squeeze(0).detach().cpu()