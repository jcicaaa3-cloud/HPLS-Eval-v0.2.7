# Security Notes

## Checkpoint loading

Use `torch.load()` only for checkpoints from a trusted source, preferably checkpoints produced by your own training pipeline.

Do not load arbitrary external `.pth` or `.pt` files in an evaluation environment without verifying origin and integrity. PyTorch checkpoint loading can execute unsafe pickle payloads when used carelessly.

## Recommended practice

- Keep SHA-256 hashes for model files, masks, predictions, compatibility CSVs, and profile logs.
- Run third-party checkpoints in an isolated environment.
- Prefer safe formats such as ONNX or safetensors when feasible.
- Record the exact exporter/runtime versions in `notes` or a separate environment log.
