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
- *Why not always take the most likely token?* Greedy decoding gets repetitive quickly. Set temperature to 0 on the Try it page to see it happen.
- *What does low probability on a token tell you?* That the model was choosing between many reasonable options there, like a character's name, as opposed to a spot where grammar forces the answer.


## Phase 6: The build page and deployment

**"How I built it" page.** Nothing on it is typed in by hand. The stats table reads `config.json`, the loss chart reads `metrics.json`, and the architecture diagram fills in layer counts and widths from the same config. Pick a different model and every number changes with it, so the page was correct for the tiny v0 model and stays correct when bigger models are added.

**Deployment.** The site is a static build pushed to a `gh-pages` branch and served by GitHub Pages:

```
cd web
npm run deploy
```

Three things made that work without a server:

- **Relative paths.** Vite's `base` is `./` and the app finds its models relative to the page, so the same build works at a domain root, in a subfolder like `/tinygpt/`, or on another static host.
- **Hash routing.** Pages are `#/compare` and `#/build`. A static host only has one real file, so there is nothing to configure for deep links.
- **Big files split at export.** GitHub rejects files over 100 MB and the base model's full-precision file is bigger than that. The export script cuts large files into 45 MB parts and lists them in `config.json`; the downloader fetches them in order and joins them.

**Checked in production.** `npm run e2e -- <url>` drives headless Chrome against the live URL at desktop and phone sizes: it waits for the model to load, generates text, opens every tab and page, and fails on any console error or sideways scroll. The `.wasm` runtime and the `.onnx` files are served correctly by GitHub Pages.

**One limit of static hosting.** Multi-threaded WebAssembly needs special HTTP headers (cross-origin isolation) that GitHub Pages can't send, so the CPU fallback runs on one thread. WebGPU is not affected.

**Questions this answers**

- *How do you host an ML demo for free?* Run the model on the visitor's device. The host only serves static files.
- *How do you avoid the page going stale when the model changes?* Every number on it is read from files the export script writes.
- *What would you change with a real backend?* Proper response headers for threaded WebAssembly, and a CDN for the model files.


## Phase 7: Real training

Same code as the tiny test model, just bigger configs and more steps. Everything ran on one RTX 3060 (12 GB) with mixed precision.

| Model | Parameters | Steps | Tokens seen | Time | Speed | Final val loss |
| --- | --- | --- | --- | --- | --- | --- |
| v0 smoke | 0.7M | 6,000 | 25M | 74 s | | 2.377 |
| small | 10.5M | 25,000 | 410M | 53 min | 134k tokens/sec | 1.342 |
| base | 27.4M | 36,000 | 590M | 2 h 47 min | 60k tokens/sec | 1.208 |

The training set is 559M tokens, so small saw about three quarters of it once and base saw all of it just over once.

**What the loss means.** Loss is the average surprise per token, in nats. `exp(loss)` is the perplexity: roughly how many tokens the model is choosing between at each step. v0 is at about 10.8, small at 3.8, base at 3.3. A model guessing at random over 4,096 tokens would be at 4,096.

**What the curve looked like.** Base's validation loss went 1.58 at step 4,000, 1.39 at 12,000, 1.30 at 20,000, 1.24 at 28,000 and 1.21 at 36,000. Fast at first, then slow steady gains. Train and validation loss stayed close the whole way, which is what you expect when the model only sees each story about once: it has no chance to memorize. It was still improving when it stopped, so more steps would have helped a little.

**Bigger was better, at a price.** Base has 2.6 times the parameters of small and ran at less than half the speed. It bought a drop from 1.342 to 1.208. The Compare page runs both on the same prompt and seed so you can read the difference for yourself.

**In the browser.** Measured in Chrome on the same desktop, 120 tokens per run:

| Model | Download (fp32 / int8) | WebGPU fp32 | WebAssembly int8 | WebAssembly fp32 |
| --- | --- | --- | --- | --- |
| small | 42 MB / 11 MB | 87 tokens/sec | 31 | 22 |
| base | 110 MB / 28 MB | 76 tokens/sec | 13 | 9 |

- On the GPU the bigger model costs almost nothing. On the CPU it is more than twice as slow, because the CPU feels every extra multiply.
- The small numbers were taken while the GPU was busy training base, so they are a bit low.
- The CPU path runs on one thread (see Phase 6), which is why it trails so far behind.

**int8 against fp32.** The int8 file is a quarter of the size and about 45% faster on the CPU. On 20 test prompts it picked the same top token as full precision every time for base, and 95% of the time for small. The raw scores do shift (up to 0.9 on a logit for base), so the probabilities are slightly different, but the stories read the same. That is why the site uses fp32 on WebGPU, where speed is not the problem, and int8 on the CPU, where download size and speed both matter.

**The base file is over GitHub's 100 MB limit.** The export script split it into three parts and the site joins them while downloading. Nothing in the web code had to change to add either model: export, a new entry in `manifest.json`, deploy.

**Questions this answers**

- *How do you know the model is not just memorizing?* Validation loss is measured on stories it never trained on, and it tracks the training loss closely.
- *What is perplexity?* `exp(loss)`. The number of equally likely choices that would give the same amount of surprise.
- *Why does a 2.6x bigger model only improve the loss by 10%?* Loss falls roughly with the log of model size and data. Each further gain costs more than the last.
- *What does quantization cost you?* Here, a 4x smaller file and faster CPU inference for a small shift in probabilities and no change in the top pick on the test prompts.
- *What would you do with more GPU time?* Train base longer first, since its curve had not gone flat. Then a bigger context window.


## Phase 8: Fine-tuning a pretrained model for comparison

The question: how does my from-scratch model compare to taking someone else's pretrained model and adapting it to the same stories?

**The model.** SmolLM2-135M, a 135M parameter model trained on general web text. I picked it over the 360M version so the browser download stays near 165 MB. This is the one place the project uses the Hugging Face libraries, and only to load the pretrained weights. The LoRA code is my own (`model/tinygpt/lora.py`).

**LoRA in plain words.** Instead of changing a big weight matrix `W`, freeze it and learn a small correction next to it: `W + B·A`, where `A` and `B` are two thin matrices (rank 8 here). `B` starts at zero, so at step 0 the model behaves exactly like the original. Only `A` and `B` get gradients.

- Applied to the query and value projections in all 30 layers, 60 matrices in total.
- 460,800 trainable parameters out of 134.5M. That is 0.34%.
- When training is done, `B·A` is added into `W` once and thrown away, so the finished model is the same size and speed as the original.

**The run.** 3,000 steps, 12.3M tokens of the same TinyStories text, 24 minutes on the same GPU.

**Results.** The two models use different tokenizers, so loss per token can't be compared directly: a model with bigger tokens has fewer, harder guesses. Dividing by the length of the text instead gives loss per byte, which is fair to both.

| Model | Trained on stories | Val loss per token | Val loss per byte |
| --- | --- | --- | --- |
| SmolLM2-135M as downloaded | nothing | 2.131 | 0.518 |
| SmolLM2-135M + LoRA | 12M tokens, 24 min | 1.666 | 0.405 |
| small (mine, 10.5M) | 410M tokens, 53 min | 1.342 | 0.337 |
| base (mine, 27.4M) | 590M tokens, 2 h 47 min | 1.208 | 0.303 |

**What I take from that**

- Fine-tuning is very efficient. Training 0.34% of the weights for 24 minutes on 2% of the data closed about half the gap between the stock model and my best one.
- On this narrow task the small specialist still wins. My 27M model predicts these stories better than a 135M model that was adapted to them, because every one of its parameters was spent on this one kind of text.
- The pretrained model knows far more. Ask it about something outside children's stories and it has an answer; mine does not. The comparison only says who is better at TinyStories.
- A longer fine-tune or a higher rank would likely narrow the gap. I stopped at 24 minutes to keep the whole project inside one afternoon of GPU time.

**In the browser.** The fine-tuned model is optional on the Compare page and nothing downloads until you ask for it. It is exported to ONNX with a key-value cache, quantized to int8 (165 MB) and run through Transformers.js in its own Web Worker. It writes about 19 tokens/sec on the CPU. My base model does 13 on the same CPU path without a cache and 76 on WebGPU.

**A bug that only showed up in production.** The panel loaded in 10 seconds on my machine and hung on the live site. GitHub Pages gzips the model files, so the size header is the zipped size. The library made a buffer that big, then rebuilt and copied the whole thing for every chunk that arrived past the end, tens of megabytes hundreds of times. The fix is in the worker: it reads each file itself and hands the library a response with the real size. Now it loads in about 14 seconds on the live site.

**Questions this answers**

- *What is LoRA and why use it?* A low-rank update trained next to frozen weights. It needs a fraction of the memory and time of full fine-tuning and the result merges back into the original weights.
- *Why start `B` at zero?* So the model's output is unchanged at the first step and training starts from the pretrained behaviour, not from noise.
- *How do you compare models with different tokenizers?* Normalize by something both share, like bytes or characters of text.
- *When would you train from scratch instead of fine-tuning?* When the task is narrow, you have plenty of data for it, and you need the model small. Otherwise fine-tuning gets most of the way for far less compute.
