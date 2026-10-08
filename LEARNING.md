# Learning notes

What I built in each phase, why it works the way it does, and the interview questions it lets me answer.

## Phase 0: Setup

Nothing clever here, but a few choices matter later:

- **One repo, two halves.** `model/` is Python and `web/` is TypeScript. The only things they share are files: a tokenizer JSON, an ONNX model, and a set of test cases in `shared/fixtures/` that both sides have to pass.
- **torch is not in the dependency list.** The right PyTorch build depends on the machine (CUDA on my desktop, CPU in CI), so you install it first and then install the package.
- **CI runs on CPU.** Every test uses tiny models and fake data, so the whole suite takes seconds and never needs a GPU or the dataset.

**Questions this answers**

- *How do you keep two implementations of the same thing in sync?* Shared golden test vectors that both test suites load.
- *Why do your tests not need a GPU?* They check behavior (shapes, masking, determinism), not model quality, and behavior shows up on a 30k-parameter model.

## Phase 1: Byte-level BPE tokenizer

A model can't read text, it reads numbers. The tokenizer is the piece that turns text into a list of token ids and back.

**How it works**

1. **Start from bytes.** Every string is UTF-8 bytes, so the first 256 token ids are just the 256 possible byte values. That alone can represent any text in any language, which is why there is never an "unknown token".
2. **Split the text into chunks first.** A regex (the same one GPT-2 uses) cuts text into words, numbers, punctuation and whitespace, with the leading space attached to the word (`" girl"`). Merges never cross a chunk boundary, so a token can't be half of one word glued to half of the next.
3. **Merge the most common pair, over and over.** Count every pair of neighboring tokens, take the most frequent pair, and give it a new id. Repeat 3,839 times. Common words end up as a single token and rare words get spelled out of pieces.
4. **Encoding replays the merges.** To encode a chunk, start from its bytes and keep applying whichever possible merge was learned earliest until none apply.
5. **Decoding is a lookup.** Each id maps to some bytes. Join them and decode as UTF-8.

The vocab is 4,096 tokens: 256 bytes + 3,839 merges + 1 special `<|endoftext|>` token that marks where one story ends.

**Making training fast enough**

The naive loop recounts every pair in the whole text for every merge. Two changes fix that:

- Count each *unique chunk* once and remember how many times it appears. TinyStories has a small vocabulary, so 50 MB of text is only a few tens of thousands of unique chunks.
- Keep a running table of pair counts and an index of which chunks contain which pair. After a merge, only the chunks that contained that pair get updated.

Encoding the full 2.2 GB dataset also leans on a cache: each unique chunk is encoded once and looked up after that. The file is read in 8 MB pieces that end on a story boundary and written straight to a `uint16` file, so memory stays flat.

**Two implementations, one answer**

The browser needs the tokenizer too, so there is a TypeScript port in `web/src/tokenizer/`. Both read the same `tokenizer.json`, and the splitting regex is stored *inside* that file so the two can't drift. `shared/fixtures/tokenizer_cases.json` has 20 tricky inputs (emoji with joiners, accents, CJK, whitespace runs, control characters, the special token) with the ids Python produced, and the TypeScript tests have to match every one exactly.

One detail that bit: `\s` means slightly different things in Python and JavaScript, so the pattern uses `\p{White_Space}`, which is defined by Unicode and is the same in both.

**Questions this answers**

- *Why BPE instead of characters or whole words?* Characters make sequences very long, so the model wastes its context window on spelling. Whole words need a huge vocab and still can't handle a word they have never seen. BPE sits in between: common words are one token, anything else falls back to pieces, and nothing is ever out of vocabulary.
- *Why byte-level?* The base alphabet is fixed at 256 and covers every language and emoji for free.
- *Why split with a regex before merging?* It keeps tokens aligned with words and makes training much cheaper, since you count unique chunks instead of scanning raw text.
- *Why a vocab of 4,096?* TinyStories uses simple English, so a small vocab covers it well. A smaller vocab also means a smaller embedding table, which matters when the whole model is a few million parameters, and ids fit in `uint16`.
- *How do you know the browser tokenizer is right?* It has to reproduce Python's ids on shared golden cases. If the two disagreed, the model would get inputs it was never trained on and quietly write worse text.


## Phase 2: The GPT model and the training loop

The model is a decoder-only transformer, the same shape as GPT-2 but much smaller. It takes a list of token ids and, for every position, outputs a score for each of the 4,096 tokens saying how likely it is to come next.

**The pieces, in the order data flows through them** (`model/tinygpt/model.py`)

1. **Token embedding.** A table with one vector per token id. This is where a token's "meaning" lives.
2. **Position embedding.** A second table with one vector per position (0 to 255). Attention has no idea about order on its own, so position has to be added in. Mine is learned, like GPT-2.
3. **Blocks**, repeated N times. Each block has two parts, and each part adds its result back onto the running vector (a residual connection):
   - **Causal self-attention.** Every token builds a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I pass along?"). The score between two tokens is `query · key / sqrt(head_dim)`. A softmax turns each row of scores into weights that add up to 1, and the output is the weighted mix of values. Several heads do this in parallel so they can look for different things.
   - **MLP.** Widen to 4x, apply GELU, narrow back. Attention moves information between positions; the MLP processes it at each position.
4. **Final LayerNorm, then the output layer**, which shares its weights with the token embedding.

**Things I can point at in the code**

- **The causal mask.** A lower-triangle matrix. Before the softmax, every score for a later position is set to `-inf`, so its weight comes out as exactly 0. Without it the model could look at the answer during training and would learn nothing useful.
- **Why divide by `sqrt(head_dim)`.** Dot products grow with vector size. Big scores push the softmax into a state where one weight is nearly 1 and the gradients vanish.
- **Pre-LayerNorm.** Normalizing *before* attention and the MLP (instead of after) leaves the residual path untouched, which makes deep stacks train more stably.
- **Weight tying.** The embedding turns ids into vectors and the output layer turns vectors back into scores over ids, so they can share one matrix. For the smoke model that matrix is more than half the parameters.
- **No shortcuts.** No `nn.Transformer`, no `nn.MultiheadAttention`, no `scaled_dot_product_attention`. The attention is written out by hand, which is also why the model can return the attention weights for the visualizations.

**The training loop** (`model/tinygpt/trainer.py`)

- **Batches** are random 256-token windows from the token file. The target is the same window shifted by one, so every position is a "predict the next token" example.
- **AdamW with selective weight decay.** Decay goes on the weight matrices only. Biases and LayerNorm parameters are left alone; shrinking them hurts and they are not where overfitting happens.
- **Warmup then cosine decay.** The learning rate ramps up over the first steps (early gradients are noisy and Adam's statistics are not settled yet), then eases down along a cosine curve.
- **Gradient clipping** at norm 1.0 so one bad batch can't throw the weights.
- **Gradient accumulation** to fake a bigger batch on a small GPU.
- **Mixed precision** (float16 autocast with a gradient scaler) on CUDA only, since that is where it pays off.
- **Checkpoints with real resume.** A checkpoint holds the model, the optimizer, the step, and every random number generator state, including the one that picks batches. A test stops a run half way, resumes it, and checks the final weights are bit-for-bit identical to a run that never stopped.
- **Metrics** go to `metrics.jsonl`: step, train loss, val loss, learning rate, tokens per second.

**Three sizes**

| Config | Params | Layers / heads / width | Context |
| --- | --- | --- | --- |
| smoke | 0.7M | 3 / 3 / 96 | 64 |
| small | 10.5M | 5 / 6 / 384 | 256 |
| base | 27.4M | 8 / 8 / 512 | 256 |

**v0 result.** The smoke model trained for 6,000 steps (24.6M tokens, 74 seconds on the GPU) and reached a validation loss of 2.38. It already writes text like: *"Once upon a time, there was a little girl named Lily. She loved to play with her friends. One day, she found a big, red ball in the park."*

**Questions this answers**

- *Why is causal masking needed?* Training feeds the whole sequence in at once and asks for a prediction at every position. Without the mask, position 5 could just read token 6. The test for this changes one later token and checks that no earlier output moves.
- *What do the query, key and value do?* Query and key decide *where* to look, value decides *what* gets copied from there.
- *Why multiple heads?* One head gives one set of weights per token. Several heads let a token attend to several things at once, for example the subject of the sentence and the previous word.
- *What does a loss of 2.38 mean?* Loss is the average negative log-probability of the right next token. Random guessing over 4,096 tokens is `ln(4096) = 8.3`. A loss of 2.38 is like choosing between about 11 equally likely tokens (`e^2.38`).
- *Why AdamW instead of Adam with L2?* In Adam, an L2 penalty gets rescaled by the adaptive step size, so it stops acting like real weight decay. AdamW applies the decay directly to the weights.
- *How do you know resume is correct?* A test compares an interrupted and resumed run against an uninterrupted one and requires identical weights.


## Phase 3: ONNX export and parity

PyTorch can't run in a browser, so the trained model is exported to ONNX, a file format that describes the model as a graph of standard operations. ONNX Runtime Web then runs that file on the visitor's device.

**What the export produces** (`model/tinygpt/export.py`)

One folder per model, which is all the website needs:

- `model.onnx`: full precision (fp32).
- `model.int8.onnx`: weights stored as 8-bit integers, about a quarter of the size.
- `tokenizer.json`, `config.json` (sizes, parameter count, training tokens, parity numbers) and `metrics.json` (the loss curve, thinned to about 200 points).

**The graph's inputs and outputs**

- Input: `input_ids`, with a sequence length that is allowed to change from call to call.
- Output 1: `logits` for the **last position only**. Generation only ever needs the next token, so the output layer runs on one vector instead of all 256.
- Output 2: `attention`, shaped (layers, heads, time, time), for the heatmap.

**No KV cache, and what that costs.** Each new token re-runs the model on the whole context. A KV cache would save each layer's keys and values so a new token only costs one position of work. That turns generation from quadratic to linear in sequence length, but it makes the graph much more complicated (every layer needs extra inputs and outputs) and it is harder to check. With a 256-token context the full pass is affordable, so I kept the simple version.

**Parity checks**

Exporting is a translation, so it gets tested like one. The same inputs go through PyTorch and ONNX Runtime at 20 different sequence lengths, including 1 and the maximum:

- fp32 has to match almost exactly. v0: largest logit difference 0.000013.
- int8 is lossy by design, so the check is "does it still pick the same top token?". v0: 100% agreement on real validation text, largest logit difference 0.62.

**A bug this caught.** The first export was 4.6 MB for a 2.9 MB model. The output layer shares the embedding matrix, but the exporter pre-computed the transpose it needed and stored it as a second copy. Multiplying by the embedding table directly keeps one copy, and the file dropped to 3.0 MB.

**Questions this answers**

- *Why ONNX?* It runs in the browser on WebGPU or WebAssembly with no server, so hosting is free and nothing a visitor types leaves their device.
- *What is quantization and what does it cost?* Storing each weight in 8 bits instead of 32. The file is about 4x smaller and CPU inference is faster, but the outputs shift a little. I measure the shift instead of assuming it is fine.
- *What is a KV cache and why skip it?* See above: a speed optimization that trades simplicity for linear-time generation.
- *How do you know the exported model is the same model?* Numeric parity against PyTorch across many sequence lengths, run as a test and again on every real export.


## Phase 4: The web app

The website is a static React app. There is no backend: the browser downloads the model file and runs it locally with ONNX Runtime Web.

**The one interface everything hangs on** (`web/src/generator/types.ts`)

```ts
interface TextGenerator {
  load(onProgress): Promise<void>
  nextLogits(ids: number[]): Promise<Float32Array>   // scores for the next token
  attention?(ids: number[]): Promise<AttentionData>  // optional
}
```

The UI only ever talks to this. Two things implement it:

- `MockGenerator`: instant and deterministic. The tests use it, and adding `?mock=1` to the URL puts it in the model picker, so the whole site works with no model files at all.
- `OnnxGenerator`: the real one. It is a thin wrapper that posts messages to a **Web Worker**, and the worker owns the ONNX session.

**Why a worker.** Running the model is heavy math. On the main thread it would freeze scrolling, the Stop button and the text streaming in. In a worker the page stays responsive and Stop works the moment you press it.

**Picking a backend.** The worker tries WebGPU with the full-precision file first. If the browser has no GPU support, or creating the session fails, it falls back to WebAssembly on the CPU with the 8-bit file, which is a quarter of the download. It runs one tiny input before reporting success, so a backend that loads but can't actually run never reaches the user. `?backend=wasm&precision=fp32` forces a combination for benchmarking.

**Sampling happens in TypeScript.** The model only returns scores. Temperature, top-k and top-p are applied in `web/src/lib/sampling.ts`, mirroring the Python version, with a small seeded random number generator so the same seed always gives the same story.

**The manifest is the only source of truth.** `public/models/manifest.json` lists the models. Each model folder describes itself in `config.json`, including which files exist and how big they are. The app has no model names or sizes in its code, so adding a model is a folder plus one manifest entry. Files over GitHub's 100 MB limit are split into parts at export time and the downloader stitches them back together.

**States that are easy to forget.** Download progress with real byte counts, a skeleton while the model starts, a clear message with a retry button when a browser can't run it, and an empty state that says what to do.

**First numbers** (v0 model, headless Chrome on my desktop): about 100 tokens/sec on WebGPU and about 190 tokens/sec on WebAssembly with the 8-bit file. For a model this small the CPU wins, because each WebGPU call has a fixed overhead that is bigger than the math itself. Bigger models flip that (see Phase 7).

**Questions this answers**

- *Why run the model in a Web Worker?* Inference blocks whatever thread it is on. The worker keeps the page responsive and makes Stop instant.
- *How does the site work before the real model exists?* Everything is written against an interface, with a mock behind it. Swapping in a trained model changes data files, not code.
- *WebGPU or WebAssembly?* WebGPU is much faster for big matrices but has per-call overhead and isn't everywhere. WebAssembly runs everywhere. The app tries the GPU and falls back.
- *How is generation reproducible in the browser?* A seeded PRNG drives the sampling, so prompt + settings + seed fully determine the output.
- *Why sample in JS instead of inside the ONNX graph?* It keeps the graph simple, lets the UI show the raw probabilities, and lets the sliders change without re-exporting anything.


## Phase 5: Looking inside the model

The output panel has four tabs. Each one shows the same generated text from a different angle.

- **Text.** The story, streaming in token by token.
- **Tokens.** Every token as a colored chip with its id. Spaces show as dots, so you can see that `" girl"` with its leading space is one token. The same token always gets the same color.
- **Probabilities.** Each generated token is shaded by how unsure the model was. Tapping one shows the top 10 candidates at that step with their probabilities. These are the model's raw probabilities, before temperature or top-k change anything.
- **Attention.** Pick a layer and a head, then tap a token. The earlier tokens light up by how much attention that token paid them. A full grid view shows every token against every other token; the empty triangle is the future that the causal mask hides.

**How the data gets there**

- The top 10 for each step is computed when the token is sampled and stored with it, so the probability view costs nothing extra.
- Attention is not fetched during generation. It is megabytes of numbers per step and nobody is looking at it yet. When the Attention tab opens, the app runs the model once more on the finished text and asks for the attention output that time.
- Models exported without attention simply don't have the method, and the tab explains that instead of breaking.

**Model comparison.** The Compare page runs one prompt through two models with the same settings and seed, one after the other so each gets the whole machine and the speed numbers are fair.

**Made for phones.** Nothing depends on hover. Tokens are tap targets, the top-10 list sits under the text on a narrow screen instead of in a floating popover, and the attention view leads with highlighted text because a 256 by 256 grid is unreadable on a phone.

**Questions this answers**

- *What does an attention head actually do?* For each token it produces a set of weights over the earlier tokens. Different heads learn different patterns, and you can flip between them to see that.
- *Why can a model pick a token that was not its top guess?* Sampling. With temperature above 0 the model rolls dice weighted by probability, which is what keeps the writing from looping.
- *Why not always take the most likely token?* Greedy decoding gets repetitive quickly. Set temperature to 0 in the playground to see it happen.
- *What does low probability on a token tell you?* That the model was choosing between many reasonable options there, like a character's name, as opposed to a spot where grammar forces the answer.
