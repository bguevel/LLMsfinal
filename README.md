# Parallel Structural Embeddings for Logic Truth Prediction

This project tests whether adding parallel structural embeddings to a small transformer improves how quickly and how accurately the model learns to predict the truth value of propositional logic statements.

The core research question is:

Can a transformer learn statement truth prediction faster, and generalize more accurately across statement complexity levels, when it receives explicit structural signals in addition to normal token and position embeddings?

The project compares several embedding modes against the same generated logic data. Each mode trains a model to read a formula such as:

```text
( A AND B ) -> A
```

and predict a single answer token:

```text
true
```

or:

```text
false
```

Training rate is tracked through answer loss over time. Accuracy is measured by checking whether the model predicts the correct truth label for statements grouped by complexity.

## What The Project Is Trying To Accomplish

The project is an experiment in giving a transformer more useful inductive bias for formal reasoning.

A regular transformer sees a logic formula as a flat sequence of tokens. It can learn the meaning of parentheses, implication, conjunction, disjunction, and variable placement from examples, but it must infer all of that structure from raw token order.

This project asks whether training improves if the model receives extra learned embedding channels that identify useful logic structure directly. These channels are "parallel" in the sense that they are computed separately from the token stream, embedded through their own embedding tables, and added into the model input alongside the token and position embeddings.

The experiment looks at two related outcomes:

1. Training rate: whether a mode reaches lower answer loss faster than the regular token-plus-position baseline.
2. Prediction accuracy: whether a mode improves true/false prediction accuracy, especially as formulas become more complex.

The menu in `main.py` supports both measurements:

- Option `0` trains every custom embedding mode sequentially and plots average answer loss by mode.
- Option `3` tests every trained mode on an input file and plots complexity vs accuracy by mode.

## Logic Task

The data consists of labeled propositional logic formulas. The supported operators are:

- `AND`
- `OR`
- `->` for implication
- parentheses for grouping
- named variables such as `A`, `B`, `P`, or `Q`

Truth labels are computed symbolically using truth tables. A statement is labeled `true` when it is a tautology, meaning it evaluates to true for every assignment of its variables. Otherwise, it is labeled `false`.

Input data is saved as tab-separated text:

```text
formula<TAB>true|false<TAB>complexity
```

Example:

```text
( A AND B ) -> A	true	Level 1
```

Generated statements can be grouped by complexity level. In the default menu flow, `data/generations.txt` controls how many statements are generated per level, how many complexity levels are used, and which variable names are available.

## Model Overview

The model is a small decoder-only transformer implemented in `custom_llm.py`.

Default configuration:

- `d_model`: 256
- `d_hidden`: 1024
- `n_heads`: 4
- `d_head`: 64
- `num_blocks`: 6
- `max_seq_len`: 4096

The model uses:

- a word-level tokenizer
- learned token embeddings
- learned positional embeddings
- optional learned structural embeddings
- causal self-attention
- RMSNorm
- an MLP block
- a final linear unembedding layer over the tokenizer vocabulary

During training, the formula is the prompt and the label is the next token. The loss is only applied to the answer token position, so the model is trained specifically to predict `true` or `false` after reading the formula.

## Embedding Modes

Every mode starts with the same base input:

```text
token embedding + position embedding
```

The experimental modes add one or more extra learned embedding vectors at each token position:

```text
token embedding
+ position embedding
+ optional structural feature embedding(s)
```

These structural features are not hard-coded answers. They are learned vectors indexed by simple feature IDs. The feature extractor tells the model what kind of structural role each token has, and the transformer learns how useful those roles are for prediction.

### `regular`

The regular mode is the baseline:

```text
token embedding + position embedding
```

This mode does not receive explicit structural hints. It must learn formula structure only from token identity and token order.

This baseline is important because every other mode is compared against it. If a structural mode trains faster or predicts more accurately than `regular`, that suggests the added structure helped.

### `depth`

Depth mode adds a parenthesis-depth embedding.

For each token, the code tracks how deeply nested the token is inside parentheses. Tokens deeper inside grouped subexpressions receive different depth IDs than top-level tokens.

This mode is looking for:

- nested subformula boundaries
- grouping depth
- whether a token belongs to an outer or inner expression
- structural complexity caused by parentheses

Example idea:

```text
( A AND ( B OR C ) )
```

The tokens inside `( B OR C )` receive a deeper nesting signal than the outer `A AND ...` expression.

The hypothesis is that truth prediction may improve if the model can more easily tell which tokens belong to the same local subformula.

### `side`

Side mode adds a left-side/right-side embedding around implication.

The feature extractor scans for `->`. Tokens before the first implication are marked as left-hand side tokens, the implication token itself gets its own ID, and tokens after it are marked as right-hand side tokens.

This mode is looking for:

- antecedent tokens
- consequent tokens
- the implication boundary
- whether a token appears before or after `->`

For a formula like:

```text
A -> B
```

`A` is marked as left-hand side, `->` is marked as the implication boundary, and `B` is marked as right-hand side.

The hypothesis is that implication-heavy truth prediction may benefit from knowing which expressions are conditions and which are conclusions.

### `logical`

Logical mode adds an embedding for the logical role of each token.

Each token is classified into a small set of categories:

- padding
- variable
- parameter or other token
- parenthesis
- `OR`
- `->`
- `AND`

This mode is looking for:

- whether a token is a variable or an operator
- which logical operator is present
- where parentheses occur
- whether the token participates as syntax or content

The model still sees the original token. The logical embedding gives it a parallel "type tag" that can help it separate formula syntax from variable names.

### `distance`

Distance mode adds an embedding based on distance to the nearest implication token.

For each token, the feature extractor finds the closest `->` and records the absolute token distance to that implication. Distances are capped by `max_implication_distance`.

This mode is looking for:

- proximity to implication boundaries
- local neighborhoods around `->`
- how far a token is from the main implication structure

The hypothesis is that implication is especially important for tautology prediction, so knowing how close each token is to an implication may help attention organize the formula.

### `all`

All mode combines every structural channel:

```text
token embedding
+ position embedding
+ parenthesis-depth embedding
+ side embedding
+ logical-role embedding
+ implication-distance embedding
```

This mode tests whether the structural features are complementary. It gives the model the richest parallel representation, but it also adds the most extra learned parameters and feature signals.

The experiment can show whether combining all structural embeddings improves performance, or whether some smaller single-feature mode is more effective.

## How The Parallel Embeddings Work

The model does not concatenate these embeddings. It adds them together.

For each token position, the model builds a vector `x`:

```text
x = token_embedding + position_embedding
```

For non-regular modes, it computes structural feature IDs for the same token sequence:

```text
depth_ids
side_ids
logical_ids
distance_ids
```

Then the selected feature IDs are passed through their own learned embedding tables:

```text
depth_emb(depth_ids)
side_emb(side_ids)
logical_emb(logical_ids)
distance_emb(distance_ids)
```

Depending on the active mode, those vectors are added to `x`.

This means every token representation remains `d_model` dimensional. The transformer architecture after the embedding step does not need to change across modes. The difference is the information placed into the residual stream before attention begins.

## How Training Works

Training data is loaded as labeled statements. Each sample becomes:

```text
prompt = formula text
answer = true or false
```

The tokenizer encodes the prompt and the one-token answer. The training batch is built so the model sees the formula and is asked to predict only the answer token.

The loss function is cross-entropy over the vocabulary, but all non-answer positions are ignored. This keeps the task focused on truth prediction rather than formula reconstruction.

Checkpoints are saved per mode:

```text
weights/custom_logic_llm_regular.pt
weights/custom_logic_llm_depth.pt
weights/custom_logic_llm_side.pt
weights/custom_logic_llm_logical.pt
weights/custom_logic_llm_distance.pt
weights/custom_logic_llm_all.pt
```

Option `0` in the menu trains all modes sequentially and writes an average loss plot to:

```text
data/average_loss_by_mode.svg
```

That plot is the main training-rate comparison.

## How Prediction And Unembedding Work

After the transformer blocks, the model applies a final RMSNorm and then a linear unembedding layer:

```text
logits = unembed(final_hidden_states)
```

The unembedding is a learned linear projection from the model dimension back to the tokenizer vocabulary size. For every position in the sequence, it produces one score, or logit, for every token in the vocabulary.

In this project, prediction only needs the final position. Given a formula prompt, the model runs forward and reads:

```text
logits_at_last_token = model(prompt_tokens)[0, -1]
```

Then it selects only the logits for the `false` and `true` vocabulary tokens:

```text
false_logit = logits_at_last_token[false_id]
true_logit = logits_at_last_token[true_id]
```

Those two logits are passed through a softmax. If `true` has the higher probability, the model predicts `true`; otherwise, it predicts `false`.

The rest of the vocabulary still exists because the model is implemented as a general next-token predictor, but evaluation narrows the decision to the two valid answer tokens.

When the tokenizer grows, both the token embedding table and the unembedding layer are resized. Existing rows are preserved, and new rows are randomly initialized.

## Evaluation

Evaluation loads a testing file, verifies the symbolic truth labels, and asks each trained mode to predict every statement.

For each statement, the evaluator records:

- expected label
- predicted label
- confidence
- complexity group
- whether the prediction was correct

It then reports accuracy per complexity level and overall accuracy.

Option `3` tests all available mode checkpoints on the same input file and writes a grouped complexity-vs-accuracy plot to:

```text
data/complexity_accuracy_by_mode.svg
```

This plot is the main accuracy comparison across formula complexity.

## Running The Project

Start the menu:

```bash
python main.py
```

Menu options:

```text
0) Train all custom embedding modes sequentially
1) Generate data
2) Train custom model
3) Test all custom embedding modes on a file and plot complexity accuracy
q) Quit
```

Typical workflow:

1. Generate a labeled statement file with option `1`.
2. Train every embedding mode with option `0`.
3. Evaluate every trained mode with option `3`.
4. Compare `data/average_loss_by_mode.svg` and `data/complexity_accuracy_by_mode.svg`.

## Key Files

- `main.py`: interactive menu, training orchestration, testing, and plotting.
- `custom_llm.py`: transformer model, tokenizer, structural embeddings, training loop, prediction, checkpointing.
- `statement_generation.py`: formula generation, parsing, labeling, saving, and loading.
- `logic.py`: propositional logic AST, truth-table evaluation, tautology checking.
- `data/generations.txt`: default statement generation settings.
- `weights/`: per-mode model checkpoints.
- `data/`: generated datasets and plots.

## Interpreting Results

A useful result is not just the highest final accuracy. The project is set up to compare several dimensions:

- Does a structural mode reduce answer loss faster than `regular`?
- Does a structural mode improve overall truth prediction accuracy?
- Does a structural mode help more on higher complexity levels?
- Does `all` outperform individual features, or does a simpler feature give cleaner gains?
- Do some structural features help training rate but not final accuracy?

The goal is to learn whether explicit parallel structural embeddings are useful for this kind of symbolic reasoning task, and which structural signals matter most.
