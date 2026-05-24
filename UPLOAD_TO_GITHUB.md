# Upload to GitHub

Recommended repository name:

```text
HPLS-Eval-v0.2.7
```

## 1. Check the folder before commit

Run from the repository root:

```bash
git status
find . -maxdepth 4 -type f | grep -Ei '(\.pem|\.key|\.env|\.pth|\.pt|\.ckpt|\.safetensors|\.bin|patient|dicom|private)' || true
```

Nothing private or heavy should appear. The repository is meant to contain the evaluation harness, templates, and documentation, not datasets, patient files, model weights, or cloud keys.

## 2. Initialize and push

```bash
git init
git branch -M main
git add .
git commit -m "Release HPLS-Eval v0.2.7"
git remote add origin https://github.com/<your-account>/HPLS-Eval-v0.2.7.git
git push -u origin main
```

## 3. Suggested GitHub description

```text
Evidence-first evaluation harness for low-precision, multi-runtime small-mask segmentation.
```

## 4. Suggested repository topics

```text
semantic-segmentation, model-evaluation, low-precision-inference, int8, onnxruntime, tensorrt, openvino, xla, runtime-benchmark, boundary-metrics
```

## 5. Release title

```text
HPLS-Eval v0.2.7 - Report polish and runtime evidence clarity
```
