# Conformal Efficiency

Minimal research codebase for comparing two training paradigms for conformal
prediction sets in multiclass classification:

1. Indirect training with cross-entropy, then exact split conformal prediction.
2. Direct training with the conformal p-value construction inside the loss,
   using the pure soft set-size surrogate.

The first implementation targets a theoretical toy setting with fixed support
points. The model is a small MLP that takes the point coordinates as input and
outputs class logits.

The repository also includes the main image benchmark:

- dataset: CIFAR-10
- model: ResNet-18 adapted for 32x32 images
- methods: indirect CE training and direct pure soft set-size training

## Quickstart

Create a run comparing both methods on the default toy experiment:

```bash
python scripts/run_experiment.py --config configs/experiment/toy_v1.yaml
```

Train only the indirect method:

```bash
python scripts/train_indirect.py --config configs/experiment/toy_v1.yaml
```

Train only the direct method:

```bash
python scripts/train_direct.py --config configs/experiment/toy_v1.yaml
```

Evaluate a saved checkpoint:

```bash
python scripts/evaluate_conformal.py \
  --config configs/experiment/toy_v1.yaml \
  --checkpoint outputs/toy_v1/indirect/seed_0/checkpoint.pt \
  --method indirect
```

Run the main CIFAR-10 benchmark:

```bash
python scripts/run_cifar10_experiment.py \
  --config configs/experiment/cifar10_resnet18_v1.yaml
```

## Project layout

```text
configs/
  data/
  experiment/
  model/
  method/
scripts/
src/conformal_efficiency/
  conformal/
  data/
  evaluation/
  models/
  objectives/
  trainers/
  utils/
outputs/
```

## Notes

- Final evaluation always uses exact randomized split-conformal p-values on the
  untouched outer calibration split.
- Direct training uses only the pure soft set-size objective. No cross-entropy
  anchor and no coverage penalty are added to that loss.
- The CIFAR-10 direct trainer refreshes inner-calibration scores once per epoch
  to keep the conformal-in-the-loop training practical.
- The existing exploratory scripts in the repository are left untouched.
