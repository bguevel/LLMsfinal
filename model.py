from __future__ import annotations

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logic import Var, And, Or, Imp, Formula
from state import ProofState
from tree_encoding import HashingTreeEmbedder, proof_state_to_tree_text


TACTICS = ["assumption", "intro", "split", "left", "right", "cases"]


TACTIC_DEFINITIONS = {
    "assumption": "Use when the goal exactly appears in the assumptions. This closes the current goal.",
    "intro": "Use when the goal is an implication A -> B. It assumes A and changes the goal to B.",
    "split": "Use when the goal is an AND statement A /\\ B. It creates two goals: prove A and prove B.",
    "left": "Use when the goal is an OR statement A \\/ B and you want to prove the left side A.",
    "right": "Use when the goal is an OR statement A \\/ B and you want to prove the right side B.",
    "cases": "Use when an assumption is an OR statement A \\/ B. It splits into two cases: assume A, and assume B.",
}


def formula_to_json(f: Formula) -> dict:
    if isinstance(f, Var):
        return {
            "type": "var",
            "name": f.name,
        }

    if isinstance(f, And):
        return {
            "type": "and",
            "left": formula_to_json(f.left),
            "right": formula_to_json(f.right),
        }

    if isinstance(f, Or):
        return {
            "type": "or",
            "left": formula_to_json(f.left),
            "right": formula_to_json(f.right),
        }

    if isinstance(f, Imp):
        return {
            "type": "imp",
            "left": formula_to_json(f.left),
            "right": formula_to_json(f.right),
        }

    raise TypeError(f"Unknown formula type: {type(f)}")


def proof_state_to_json(state: ProofState) -> dict:
    if state.is_solved():
        return {
            "status": "solved",
            "goals": [],
        }

    goals = []

    for g in state.goals:
        goals.append(
            {
                "assumptions": [formula_to_json(a) for a in g.assumptions],
                "target": formula_to_json(g.target),
            }
        )

    return {
        "status": "unsolved",
        "goals": goals,
        "active_goal_index": 0,
        "active_goal": goals[0],
    }


def proof_state_to_json_text(state: ProofState) -> str:
    return json.dumps(proof_state_to_json(state), indent=2)


def tactic_definitions_text() -> str:
    lines = []
    for tactic in TACTICS:
        lines.append(f"- {tactic}: {TACTIC_DEFINITIONS[tactic]}")
    return "\n".join(lines)


def clean_tactic_name(text: str) -> str:
    text = text.strip()
    text = text.splitlines()[0].strip()
    text = text.split(",")[0].strip()
    text = text.split()[0].strip(".,:;[]{}()\"'")

    return text


def parse_ranked_tactics(response: str) -> list[str]:
    response = response.replace("\n", ",")
    raw_items = response.split(",")

    ordered = []

    for item in raw_items:
        tactic = clean_tactic_name(item)
        if tactic in TACTICS and tactic not in ordered:
            ordered.append(tactic)

    for tactic in TACTICS:
        if tactic not in ordered:
            ordered.append(tactic)

    return ordered


class PhiTacticModel:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        include_tree_in_prompt: bool = True,
        concatenate_tree_embedding: bool = True,
        tree_embedding_dim: int = 256,
    ):
        self.model_name = model_name
        self.include_tree_in_prompt = include_tree_in_prompt
        self.concatenate_tree_embedding = concatenate_tree_embedding
        self.tree_embedder = HashingTreeEmbedder(dim=tree_embedding_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        self.model.config.use_cache = False
        self.model.eval()

    def build_tree_prompt_block(self, state: ProofState) -> str:
        if not self.include_tree_in_prompt:
            return ""

        return (
            "PROOF STATE TREE ENCODINGS:\n"
            f"{proof_state_to_tree_text(state)}\n\n"
        )

    def build_prompt(self, state: ProofState) -> str:
        return (
            "<|system|>\n"
            "You are a proof-search policy model. "
            "Your job is to choose the best next proof tactic. "
            "You must use the tactic definitions exactly as given. "
            "Return only one tactic name and no explanation.\n"
            "<|end|>\n"
            "<|user|>\n"
            "TACTIC DEFINITIONS:\n"
            f"{tactic_definitions_text()}\n\n"
            "PROOF STATE JSON:\n"
            f"{proof_state_to_json_text(state)}\n\n"
            f"{self.build_tree_prompt_block(state)}"
            f"Allowed tactics: {', '.join(TACTICS)}\n"
            "Best next tactic:\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )

    def build_ranking_prompt(self, state: ProofState) -> str:
        return (
            "<|system|>\n"
            "You are a proof-search policy model. "
            "Your job is to rank proof tactics from best to worst for the current proof state. "
            "You must use the tactic definitions exactly as given. "
            "Return only comma-separated tactic names and no explanation.\n"
            "<|end|>\n"
            "<|user|>\n"
            "TACTIC DEFINITIONS:\n"
            f"{tactic_definitions_text()}\n\n"
            "PROOF STATE JSON:\n"
            f"{proof_state_to_json_text(state)}\n\n"
            f"{self.build_tree_prompt_block(state)}"
            f"Allowed tactics: {', '.join(TACTICS)}\n"
            "Ranked tactics:\n"
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
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        tactic = clean_tactic_name(response)

        if tactic not in TACTICS:
            return "assumption"

        return tactic

    @torch.no_grad()
    def predict_tactic_order(self, state: ProofState) -> list[str]:
        prompt = self.build_ranking_prompt(state)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            use_cache=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return parse_ranked_tactics(response)

    @torch.no_grad()
    def get_tree_embedding(self, state: ProofState) -> torch.Tensor:
        return self.tree_embedder.encode_state(state)

    @torch.no_grad()
    def get_state_embedding(self, state: ProofState) -> torch.Tensor:
        prompt = self.build_ranking_prompt(state)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        llm_vec = last_hidden[:, -1, :].squeeze(0).detach().cpu()

        if not self.concatenate_tree_embedding:
            return llm_vec

        tree_vec = self.get_tree_embedding(state)
        return torch.cat([llm_vec, tree_vec], dim=0)
