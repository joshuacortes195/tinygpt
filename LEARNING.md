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
