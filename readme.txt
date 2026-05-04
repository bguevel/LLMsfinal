Current state:
This project is now a propositional-logic sandbox for testing whether explicit AST/tree structure helps a model evaluate
whether logical statements are true under every truth assignment.

The main documentation now lives in README.md.

How to run:
python main.py

Quick checks:
python test_tree_encoding.py

Generate a saved truth-evaluation file:
python statement_generation.py 100 data/generated_truth_eval.jsonl --seed 1 --max-depth 3

Default model weights are saved under:
weights/
