# Logic Truth Evaluation With AST Embeddings

## Project Goal

This project explores whether a language model can evaluate the truth of propositional logic statements more reliably when the model receives explicit tree structure in addition to text.

The central task is:

```text
Given a logical formula F, predict whether F is true or false.
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
1) Custom model, logic-aware tokenizer, regular token embeddings only.
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
0) Train custom statement model, then truth classifier, then codec encoder
1) Generate true/false statements to a file
2) Check labels in a saved statement file
3) Train embedding models
4) Train statement model
5) Run a trained model on a statement file
6) Advanced imported/custom LLM menu
```

The project is now focused entirely on statement truth evaluation.

Option 0 runs the full sequence in a fixed order: train the custom statement model first, then train the truth-classifier AST embedding, then train the AST codec encoder on the same selected JSONL lines.

Between phases, the menu releases training resources and clears CUDA cache when torch reports a CUDA device. Training errors are caught at phase boundaries so the overnight sequence can continue where possible instead of exiting the whole program.

For option 4, the flow is intentionally direct: choose the model first, then choose the embedding source, then enter the training file, line count, epochs per line, learning rate, and save path. Option 6 keeps the fuller imported/custom model menus for running checkpoints and non-statement training paths.

Inside option 4, option 0 provides the same combined custom-model training sequence.

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

`generate_and_save_labeled_statement_batches(...)` writes one JSONL file from several complexity levels, each with its own statement count.

The interactive generator reads its default generation plan from:

```text
data/generations.txt
```

Expected format:

```text
number of true and false (n): 500
levels of complexity (for each level n statements are generated): 10
set of variables: {Q, W, E, R, T, Y, U, I, O, P}
```

The variable set may span multiple lines and may include individual letters or word-like variable names:

```text
set of variables: {A, B, C,
active, valid, atLeastOne, conditionA, theorem, proof}
```

Variable names must start with a letter and may contain letters, digits, or underscores. This creates levels 1 through 10, using each level number as the random formula max depth, and generates `n` total statements at each level.

When the configured variable vocabulary is large, each individual generated formula samples a small variable subset before symbolic truth checking. This lets the full dataset use many letters and words while keeping each truth-table evaluation tractable.

During generation, each candidate formula is evaluated with the symbolic tautology checker before it is accepted for its assigned true/false label. The saved JSONL also stores `is_tautology` metadata. Training paths then trust that generated label metadata instead of recomputing symbolic truth for every example.

You can optionally add:

```text
true fraction: 0.5
```

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

## How True/False Training Data Is Validated

The training data is not trusted just because the generator intended a formula to be true or false. Every saved statement is checked symbolically before it is written to disk.

The label definition is:

```text
label = true   means formula is a tautology
label = false  means formula is not a tautology
```

This project does not use "false" to mean "false under every assignment." A false-labeled example may still be satisfiable. It only means the formula fails to be true under every possible assignment.

The validation pipeline is:

```text
1. Choose a target label.
2. Build a candidate formula.
3. Enumerate the variables that appear in that formula.
4. Enumerate every truth assignment for those variables.
5. Evaluate the formula under each assignment.
6. Accept the formula only if the symbolic result matches the target label.
7. Save the label and the symbolic result as JSON metadata.
```

For a formula `F`, the truth checker computes:

```text
vars(F) = sorted distinct variable names in F
assignments(F) = all boolean assignments over vars(F)
is_tautology(F) = all(evaluate_formula(F, assignment) for assignment in assignments(F))
```

If `F` has `k` distinct variables, the validation step checks:

```text
2^k
```

assignments. Because this cost is exponential, the generator samples a small variable subset for each individual formula when the configured variable vocabulary is large. The overall dataset can still use hundreds of variable names across examples, but one formula stays small enough to validate exactly.

True examples are built from tautology-preserving templates such as:

```text
A -> A
(A /\ B) -> A
A -> (A \/ B)
(A /\ (A -> B)) -> B
```

False examples are built from formulas that are usually not tautologies, such as a lone variable, conjunctions, and random implications. Even so, the generator does not rely on "usually." It still calls the exact tautology checker before accepting the label.

Before a record is saved, `statement_to_record(...)` calls:

```text
verified_statement_label(statement.formula, statement.label, statement.name)
```

That function recomputes symbolic truth and raises an error if the saved label would be wrong. The JSONL record then stores both:

```text
label
is_tautology
```

During model training, the code loads generated JSONL with fast metadata checks instead of recomputing full truth tables for every line. This is deliberate: symbolic validation has already happened during generation, so training can avoid paying the exponential validation cost again. If you want to audit a file manually, menu option 2 recomputes symbolic labels for the saved file.

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

Inside formula prompts, word variables are also tokenized as variables, including words that would otherwise look like labels or prompt markers:

```text
true -> VAR_true
statement -> VAR_statement
atLeastOne -> VAR_atLeastOne
```

This helps the text channel and AST channel agree about structure while keeping answer labels like `true` and `false` separate from formula variables.

The custom model can choose either tokenizer in `main.py`.

For the project's main custom-model training path, `main.py` now uses the logic-aware tokenizer by default so the tokenizer is not the experimental variable.

### Logic-Aware Tokenizer Details

The custom statement model sees a prompt, not the raw JSON record. For logic-aware training, the prompt has this shape:

```text
<STATEMENT>
{formula}
<QUESTION> tautology ?
<ANSWER>:
```

The saved JSON label is not included in that prompt. The statement name is also not included. The model sees the formula text and must predict the answer token.

The tokenizer first rewrites logical syntax into stable special tokens:

```text
(      -> <LPAREN>
)      -> <RPAREN>
/\     -> <AND>
\/     -> <OR>
->     -> <IMP>
```

Then it scans identifiers. Between `<STATEMENT>` and `<QUESTION>`, identifiers are treated as formula variables:

```text
P            -> VAR_P
active       -> VAR_active
atLeastOne   -> VAR_atLeastOne
conditionA   -> VAR_conditionA
true         -> VAR_true
false        -> VAR_false
statement    -> VAR_statement
```

This formula mode is important. Outside the formula, `true` and `false` are answer labels. Inside the formula, they may be ordinary propositional variables from `generations.txt`.

Outside formula mode, label words are mapped to answer tokens:

```text
true  -> <TRUE>
false -> <FALSE>
```

That means the tokenizer can distinguish:

```text
formula variable named true = VAR_true
answer label true          = <TRUE>
```

For labeled statement training, the model input contains the prompt tokens only. The target is the single answer token. Internally, training appends the answer token only to create the shifted causal-LM target, then masks out every target position except the answer position. In effect:

```text
input visible to model:  prompt tokens
target to predict:       <TRUE> or <FALSE>
loss applied to:         answer token only
loss ignored for:        prompt tokens and padding
```

This prevents the model from seeing the label in the input while still allowing normal causal-LM backpropagation through the token embeddings, position embeddings, attention blocks, MLP blocks, and unembedding layer.

The tokenizer vocabulary is grown dynamically from the training prompts and answer tokens. When new tokens are added, the custom model resizes its embedding table and output unembedding layer, preserving existing weights and randomly initializing rows for new tokens.

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

In the custom model menu, the "regular token embeddings only" option trains on the labeled statement JSONL with the same true/false answer objective, but passes `use_ast=False`. That means the model uses token embeddings plus position embeddings, without a parallel AST soft-prefix.

## Detailed Embedding Mechanics

There are two different meanings of "embedding" in this project:

```text
token embeddings = learned vectors for tokenizer tokens
AST embeddings   = structural vectors computed from the formula tree
```

The regular token path is always:

```text
tokens = tokenizer(prompt)
E_tok = token_embedding(tokens)
E_pos = position_embedding(positions)
X = E_tok + E_pos
```

With no AST embedding, the Transformer receives only:

```text
X
```

With AST conditioning enabled, the formula is also encoded as a tree:

```text
z = AST_encoder(formula)
```

That vector `z` is not simply concatenated to the final hidden state. Instead, it is projected into virtual prefix token embeddings:

```text
P = reshape(W_project z, tree_prefix_tokens, d_model)
```

The Transformer receives:

```text
[P; X]
```

where `[P; X]` means the AST prefix tokens are placed before the ordinary text tokens in the sequence.

This matters because self-attention is computed over the combined sequence. Text tokens can attend to the AST prefix at every layer:

```text
attention_input = [AST_prefix_tokens; prompt_tokens]
```

So the structural embedding can influence attention patterns throughout the model, not only a final classifier layer. In the custom model, after the Transformer blocks run, the prefix positions are removed before the unembedding step so the model still produces logits aligned with the original text token positions.

The custom model's AST path is:

```text
formula JSON / Formula object
        |
        v
AST encoder or hash embedder
        |
        v
tree vector z
        |
        v
linear projection W_project
        |
        v
virtual prefix token embeddings
        |
        v
Transformer attention together with token embeddings
```

The imported HuggingFace path uses the same idea through `inputs_embeds`:

```text
token_embeddings = imported_model.get_input_embeddings()(input_ids)
prefix_embeddings = W_project z
inputs_embeds = concat(prefix_embeddings, token_embeddings)
```

The imported model's tokenizer and core weights stay pretrained unless the chosen training mode says to tune more of them.

### Fixed Hash AST Embedding

The fixed hash embedding is deterministic and untrained. It extracts structural features from the tree and hashes them into a vector. Examples of features include:

```text
operator kind at a tree path
variable name at a tree path
parent-child relationships
prefix-token unigrams
prefix-token bigrams
tree size and depth statistics
```

This gives the LLM a repeatable structural signal without learning an AST encoder first. It is useful as a cheap baseline because any improvement over the regular-token-only model cannot come from training a separate tree encoder.

### Recursive AST Encoder

The recursive encoder computes a vector bottom-up.

For a variable node:

```text
z_var = tanh(W_leaf [kind(var); bucket(variable_name)] + b_leaf)
```

The variable name is placed into a stable bucket so arbitrary variable words can be embedded without requiring a separate learned row for every possible name.

For a binary node:

```text
z_node = tanh(W_bin [kind(operator); z_left; z_right] + b_bin)
```

This recursively compresses the full formula tree into one vector. The same mechanism handles:

```text
and
or
imp
```

When this encoder is untrained, it is mostly a control condition. When it is trained as part of the truth classifier or codec, the resulting encoder can be reused as the AST embedding source for the LLM.

### Truth-Classifier AST Encoder

The truth-classifier encoder is trained directly on the true/false task:

```text
z = FormulaTreeEncoder(F)
logits = linear(z)
loss = cross_entropy(logits, label)
```

After training, only the encoder side is reused for LLM conditioning:

```text
F -> z
```

This embedding is encouraged to preserve information useful for tautology classification. It may discard details that are irrelevant to the true/false decision.

### Codec AST Encoder

The codec encoder is trained with a reconstruction objective:

```text
z = FormulaTreeEncoder(F)
decoder tries to reconstruct prefix_tokens(F)
loss = cross_entropy(reconstructed_tokens, actual_prefix_tokens)
```

This embedding is encouraged to preserve enough information to rebuild the tree. It is not directly trained to predict true/false labels, but it may produce a more generally structural representation.

The codec decoder is used only during codec training and diagnostics. When conditioning the LLM, the project uses:

```text
codec.encoder(F) -> z
```

not the decoder output.

### What Backpropagation Updates

For the custom statement model, the answer-token loss backpropagates through all trainable custom-model layers:

```text
token embeddings
position embeddings
attention layers
MLP layers
final layer norm
unembedding / output head
AST projector, if AST is enabled
selected AST encoder, only if joint AST-encoder training is enabled
```

For regular-token-only training, the AST path is disabled:

```text
use_ast = False
```

so the comparison is clean:

```text
regular-only model: prompt tokens -> true/false
AST-conditioned model: AST prefix + prompt tokens -> true/false
```

For imported models, the training mode controls what can update:

```text
adapter_only                 = AST projector only
adapter_and_embeddings       = AST projector + token embeddings
adapter_and_unembedding      = AST projector + output head
full                         = imported model + projector
```

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

The labeled-statement training menu asks how many JSONL lines to train on and how many epochs to run per line. Press Enter at the line-count prompt to use the full file. For this path, an epoch means one optimizer update on the current line. Training uses one JSONL line at a time, repeats that same line for the requested epochs, then moves to the next line. Oversized or failed lines are skipped with a warning so training can continue. Progress prints every 50 line-epochs, plus once at the end.

For true/false statement data, the loss is applied to the answer token only (`true` or `false`, represented as `<TRUE>` or `<FALSE>` by the logic-aware tokenizer). The model input contains the formula prompt and optional AST prefix, but not the statement name, saved JSON label, or answer token.

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

In the interactive menu, option 1 reads `data/generations.txt`, then prompts only for the output JSONL file and optional random seed.

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
