Current state:
This project is now a small propositional-logic sandbox for exploring whether a model can evaluate the truth of logical
statements better when it sees both text and explicit formula-tree structure.

The formulas are already AST-like Python objects: Var, And, Or, and Imp. The new tree_encoding.py module adds several
ways to embed and unembed that tree structure:

1) Prefix tokens:
   Compact sequence form. Example: ["imp", "var", "P", "var", "P"].
   Good for normal transformers and exactly reversible with prefix_tokens_to_formula.

2) S-expressions:
   Human-readable tree text. Example: (imp (var "P") (var "P")).
   Good for prompts and exactly reversible with sexpr_to_formula.

3) Path entries:
   Node-at-path records such as root=imp, root.L=var(P), root.R=var(P).
   Good for tree/graph models and exactly reversible with path_entries_to_formula.

4) Hashing tree embedding:
   A fixed-size deterministic vector over node paths, operator edges, prefix ngrams, variables, and tree stats.
   Good as an immediate parallel structural embedding because it does not need training.

5) Trainable recursive tree encoder:
   A neural encoder skeleton that recursively combines operator and child embeddings.
   Good for future learned experiments. Decoding from its continuous vector needs a trained decoder head, so exact
   unembedding should use the discrete codecs above.

goals:
1) Test whether explicit AST/tree structure helps a model decide whether a logical statement is valid.

2) Compare raw proof-state JSON against JSON + tree encodings + parallel structural embeddings.

3) Keep the tree representations exactly reversible so experiments can be inspected and debugged.

How to run the stuff I have:
press run on main.py

Quick checks:
python test_tree_encoding.py

Generate a saved truth-evaluation file:
python statement_generation.py 100 data/generated_truth_eval.jsonl --seed 1 --max-depth 3

Useful generation functions:
- create_true_statement(...)
- create_false_statement(...)
- check_statement_truth(formula)
- generate_labeled_statements(n, ...)
- generate_and_save_labeled_statements(n, output_path, ...)
- load_labeled_statements(path)
