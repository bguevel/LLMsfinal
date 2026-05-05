# Logic Truth Evaluation With AST Embeddings

## Project Goal

This project explores whether a language model can evaluate the truth of propositional logic statements more reliably when the model receives explicit tree structure in addition to text.

The central task is:

```text
Given a logical formula F, predict whether F is true under every truth assignment.
```

In classical logic terms, the label is:

```text
label(F) = 1 if F is a tautology
label(F) = 0 otherwise
```

The current framework supports formulas built from:

```text
Var(name)
And(left, right)
Or(left, right)
Imp(left, right)
```

Those objects already form an AST, or abstract syntax tree. The project now uses that AST directly for generation, checking, tokenization, embedding, and model conditioning.

## Current Experiment Plan

The immediate experiments are designed to test whether explicit AST structure improves truth evaluation compared with text-only baselines.

The main comparisons are:

```text
1) Custom model, logic-aware tokenizer, no AST baseline.
2) Custom model, logic-aware tokenizer, hash AST.
3) Custom model, logic-aware tokenizer, trained AST encoder.
4) Imported model with prompt/tree text only as a baseline.
5) Imported model with adapter_only AST soft-prefix.
```

The goal is not just to see which model can memorize generated examples. The important question is whether AST structure improves performance on held-out logical statements.

## How To Run

Start the interactive menu:

```powershell
python main.py
```

Useful menu paths:

```text
1) Generate true/false statements to a file
2) Check labels in a saved statement file
3) Train the AST embedding truth model
4) Train the AST embedding/unembedding codec
5) Use the trained AST embedding truth model
6) Choose imported or custom LLM
7) Optional proof-search BFS mode
```

The main research path is statement truth evaluation. The proof-search BFS code remains available, but it is no longer the default project direction.

Model checkpoints are saved under:

```text
weights/
```

Custom-model and imported-model weights use different default filenames so the two paths do not overwrite each other:

```text
weights/custom_logic_llm.pt
weights/custom_logic_llm_text.pt
weights/custom_logic_llm_wikipedia.pt
weights/imported_ast_adapter.pt
weights/truth_embedding.pt
weights/tree_codec.pt
```

## Core Logic Functions

### logic.py

`evaluate_formula(formula, assignment)` evaluates a formula under a single truth assignment.

For implication, the semantics are:

```text
eval(A -> B) = (not eval(A)) or eval(B)
```

`truth_table(formula)` enumerates all assignments for the variables in the formula.

For variables `v_1, ..., v_k`, there are:

```text
2^k
```

truth assignments.

`is_tautology(formula)` checks whether the formula is true under every assignment:

```text
is_tautology(F) = all(eval(F, a) for a in assignments(vars(F)))
```

`formula_to_dict(formula)` and `formula_from_dict(data)` save and reload formulas as ordinary JSON.

### statement_generation.py

`create_true_statement(...)` creates a formula that should be tautologically true.

`create_false_statement(...)` creates a formula that is not a tautology.

`check_statement_truth(formula)` checks the true/false label using `is_tautology`.

`generate_labeled_statements(n, ...)` creates `n` labeled examples in memory.

`generate_and_save_labeled_statements(n, output_path, ...)` creates `n` labeled examples and writes them as JSONL.

Statement generation supports named complexity presets:

```text
simple   = shorter formulas over P, Q
moderate = default nested formulas over P, Q, R, S
complex  = deeper formulas over P, Q, R, S, U, V
```

Each saved record includes:

```text
name
label
is_tautology
is_satisfiable
text
formula JSON
tree encodings
```

### tree_encoding.py

This file contains AST encodings and structural embedding tools.

`formula_to_prefix_tokens(formula)` converts a formula to reversible prefix tokens:

```text
Imp(P, P) -> ["imp", "var", "P", "var", "P"]
```

`prefix_tokens_to_formula(tokens)` reconstructs the formula.

`formula_to_sexpr(formula)` gives a readable tree representation:

```text
(imp (var "P") (var "P"))
```

`formula_to_path_entries(formula)` records nodes by tree path:

```text
root = imp
root.L = var(P)
root.R = var(P)
```

`HashingTreeEmbedder` creates an untrained fixed vector from structural features.

`FormulaTreeEncoder` is a trainable recursive AST encoder.

### tree_codec.py

This file trains an embedding/unembedding codec for formulas.

The encoder maps an AST to a vector:

```text
z = encoder(F)
```

The decoder learns to reconstruct prefix tokens from `z`:

```text
p(t_i | t_<i, z) = softmax(W h_i + b)
```

The reconstruction loss is cross entropy over prefix tokens:

```text
L_codec = - sum_i log p(t_i | t_<i, z)
```

The trained encoder can then be used as a structural embedding source for either the custom model or the imported model.

### truth_model.py

This file trains a direct AST truth classifier.

The model is:

```text
z = FormulaTreeEncoder(F)
logits = W z + b
p(label | F) = softmax(logits)
```

The training loss is:

```text
L_truth = - log p(y | F)
```

This is not an LLM. It is a focused structural baseline for asking: can the tree alone predict tautology labels?

## Tokenization Schemes

### Simple Word Tokenizer

The simple tokenizer in `custom_llm.py` lowercases text and splits words and punctuation. It is easy to use, but logical operators may be split in ways that are not ideal.

For example:

```text
(P -> (Q /\ P))
```

may be seen as ordinary punctuation and words rather than stable logical symbols.

### Logic-Aware Tokenizer

The logic-aware tokenizer is designed for the custom LLM.

It converts formula syntax into stable logic tokens:

```text
(P -> (Q /\ P))
```

becomes:

```text
<LPAREN> VAR_P <IMP> <LPAREN> VAR_Q <AND> VAR_P <RPAREN> <RPAREN>
```

This helps the text channel and AST channel agree about structure.

The custom model can choose either tokenizer in `main.py`.

### Imported Model Tokenizer

Imported HuggingFace models should usually keep their pretrained tokenizer. Their embedding matrix and unembedding / language-model head were trained for that tokenizer.

For the imported model, the project offers:

```text
1) pretrained tokenizer
2) pretrained tokenizer + added logic special tokens
```

Full tokenizer replacement is intentionally not offered for the imported model, because replacing the tokenizer would make the pretrained token embeddings and output head poorly aligned.

## Embedding Options

Both the custom model and imported model can choose among these structural inputs:

```text
0) No AST embedding
1) Fixed hash tree embedding
2) Untrained recursive tree encoder
3) Trained truth-classifier encoder
4) Trained embedding/unembedding codec encoder
```

### No AST Embedding

This is the text-only baseline.

The model receives only tokenized prompt text:

```text
x = token_embedding(tokens) + position_embedding
```

This is important for measuring whether AST structure helps.

### Fixed Hash Tree Embedding

`HashingTreeEmbedder` creates a deterministic vector from hand-built tree features:

```text
h(F) in R^d
```

Features include:

```text
node kind at path
variable at path
parent-child operator edges
prefix token unigrams and bigrams
tree statistics
```

The vector is not trained. It is useful as a cheap structural baseline.

### Untrained Recursive Tree Encoder

`FormulaTreeEncoder` can be used before training. Its weights are random, so this is mostly a control condition.

For a variable:

```text
z_var = tanh(W_leaf [kind(var); var_bucket(name)] + b_leaf)
```

For a binary operator:

```text
z_op = tanh(W_bin [kind(op); z_left; z_right] + b_bin)
```

where `op` is one of:

```text
and, or, imp
```

### Trained Truth-Classifier Encoder

This uses the encoder from `truth_model.py` after it has been trained to predict tautology labels.

It learns tree embeddings useful for:

```text
F -> true/false
```

### Trained Embedding/Unembedding Codec Encoder

This uses the encoder from `tree_codec.py` after the codec learns to reconstruct AST prefix tokens.

It learns embeddings that preserve enough information for reconstruction:

```text
AST -> z -> prefix tokens
```

The LLMs use the encoder side:

```text
AST -> z
```

The decoder side is useful for checking whether `z` retained tree structure.

## How AST Embeddings Affect Attention

Both the custom model and imported model use AST soft-prefix tokens.

Given a tree vector `z`, the model projects it into `m` virtual token embeddings:

```text
P = reshape(W_p z, m, d_model)
```

The actual token embeddings are:

```text
X = token_embedding(tokens) + position_embedding
```

The model attends over:

```text
[P; X]
```

This matters because attention can now directly attend to the structural prefix. A vector concatenated after the forward pass would not affect attention heads.

The AST prefix is added alongside the regular token embeddings. It does not replace the ordinary embedding table:

```text
regular token stream = token_embedding(tokens) + position_embedding
structural prefix = reshape(W_p z, m, d_model)
model input = concat(structural prefix, regular token stream)
```

For the no-AST baseline, the structural prefix is omitted.

## Custom LLM Path

`custom_llm.py` adapts the user's LLM3-style Transformer.

It supports:

```text
text-only training
AST soft-prefix training
simple tokenizer
logic-aware tokenizer
hash, untrained, or trained AST encoders
optional joint training of the selected AST encoder
batched training
plain text file training
Wikipedia article training from a title/URL file
mixed plain text plus Wikipedia training
```

The custom model is flexible because its tokenizer, embeddings, attention blocks, and unembedding are all controlled in this repo.

Custom LLM training on labeled logic statements trains the whole custom model by default:

```text
token embeddings
position embeddings
attention blocks
MLP blocks
final unembedding
AST projector, when AST is enabled
selected trainable AST encoder, if joint training is selected
```

Custom text training is also available. A plain text file trains the custom model with a next-token objective:

```text
L_text = - sum_t log p(x_t | x_<t)
```

Wikipedia training expects a text file containing one Wikipedia title or Wikipedia URL per line. The code fetches plaintext article extracts through the Wikipedia API, chunks the text into sequences, and trains the custom model in batches. A ready-to-use seed list with 500 titles is included at:

```text
data/wikipedia_titles_500.txt
```

The custom model menu lets you train from a plain text file, from the Wikipedia title list, or from both sources in one run. Each path prompts for the number of epochs before training starts.

Example title file:

```text
Propositional calculus
Mathematical logic
https://en.wikipedia.org/wiki/Truth_table
```

## Imported Model Path

`imported_llm.py` wraps a HuggingFace causal language model.

It supports AST conditioning through `inputs_embeds`, using soft-prefix tokens:

```text
inputs_embeds = concat(AST_prefix_embeddings, token_embeddings)
```

Training modes:

```text
adapter_only
adapter_and_embeddings
adapter_and_unembedding
full
```

`adapter_only` freezes the imported model and trains only the AST projector.

`adapter_and_embeddings` trains the imported model's token embedding table as well. This can be useful when using the augmented logic tokenizer, because newly added tokens such as `<IMP>` and `VAR_P` begin with untrained embeddings. It is less necessary when using the pretrained tokenizer unchanged.

Imported-model checkpoints are saved separately from custom-model checkpoints. The default imported checkpoint is:

```text
weights/imported_ast_adapter.pt
```

## Limitations Of The Imported Model

The imported model is powerful but less flexible for this project.

Key limitations:

1. The tokenizer is tied to pretrained weights.

The input embedding matrix and output unembedding head were trained for the original tokenizer. Replacing the tokenizer would mostly invalidate those weights.

2. Added logic tokens start untrained.

Augmenting the tokenizer with tokens like `<IMP>` or `VAR_P` is safer than replacement, but those new token embeddings need tuning.

3. Soft-prefix conditioning does not rewrite the model's internal algorithm.

AST prefix tokens give attention heads access to structure, but the model still must learn how to use that structure.

4. Full fine-tuning is expensive.

Training the whole imported model uses more memory and compute. Adapter-only training is cheaper, but less expressive.

5. The AST codec decoder is not directly used by the imported model.

The imported model consumes the encoder vector as a soft-prefix. The codec decoder is for reconstruction and diagnostics unless you explicitly build a generation objective around it later.

6. The imported model's answer format may be brittle.

Even when the structural signal helps, a causal LM may output text other than exactly `true` or `false`. A classifier head or constrained decoding could make evaluation cleaner.

## Suggested Experiments

Start with a generated dataset:

```powershell
python statement_generation.py 200 data/generated_truth_eval.jsonl --seed 1 --complexity moderate
```

In the interactive menu, option 1 prompts for the statement complexity before writing the training or testing JSONL file. Use `Custom` if you want to enter a specific formula depth and variable list.

Then compare:

```text
custom model, simple tokenizer, no AST
custom model, logic tokenizer, no AST
custom model, logic tokenizer, hash AST
custom model, logic tokenizer, trained truth encoder
custom model, logic tokenizer, trained codec encoder
imported model, pretrained tokenizer, no AST
imported model, pretrained tokenizer, AST soft-prefix
imported model, augmented logic tokenizer, AST soft-prefix
```

The most important comparison is not whether a model can memorize generated examples. It is whether performance improves on held-out formulas when AST structure is available.
