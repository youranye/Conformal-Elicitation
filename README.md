# Conformal Prediction

Research codebase for comparing four multiclass prediction-set methods under a
common evaluation pipeline.

## Methods

Let the classifier output $p_\theta(y \mid x)$, and let labels be sorted so
$p_\theta(y_{(1)} \mid x) \ge \cdots \ge p_\theta(y_{(K)} \mid x)$.

### 1. Indirect + conformal
Train with cross-entropy, optionally apply post-hoc temperature scaling, then
form exact randomized split-conformal sets with score

$$
S_\theta(x,y)=p_\theta(y \mid x).
$$

For calibration scores $S_i^\theta=S_\theta(X_i,Y_i)$, the conformal p-value is

$$
p(x,y)=\frac{\sum_i \mathbf{1}\{S_i^\theta < S_\theta(x,y)\}
+ U\bigl(1+\sum_i \mathbf{1}\{S_i^\theta=S_\theta(x,y)\}\bigr)}{n+1},
$$

and the set is

$$
C^\alpha(x)=\{y : p(x,y)>\alpha\}.
$$

### 2. Direct + conformal
Train the score function end-to-end using the pure soft set-size surrogate
$\tilde L_{\text{size}}$, with conformal calibration inside the training
objective. Final evaluation still uses the exact conformal set

$$
C^\alpha(x)=\{y : p(x,y)>\alpha\}.
$$

### 3. Indirect + greedy cumulative mass, fixed $\tau$
Train indirectly, then build the set by accumulating top-ranked label mass
until $\tau = 1 - \alpha$:

$$
L_\tau(x)=\min\left\{\ell :
\sum_{j=1}^{\ell} p_\theta(y_{(j)} \mid x)\ge \tau \right\},
\qquad
C_{\theta,\tau}^{\text{mass}}(x)=\{y_{(1)},\dots,y_{(L_\tau(x))}\}.
$$

### 4. Indirect + greedy cumulative mass, calibrated $\hat{\tau}$
Train indirectly, then choose the smallest $\hat{\tau}$ on a calibration set
such that empirical coverage reaches the target:

$$
\hat{\tau}
=
\inf\left\{\tau :
\frac{1}{n}\sum_{i=1}^n \mathbf{1}\{Y_i \in C_{\theta,\tau}^{\text{mass}}(X_i)\}
\ge 1 - \alpha \right\}.
$$
# Conformal Prediction

Research codebase for comparing four multiclass prediction-set methods under a
common evaluation pipeline.

## Methods

Let the classifier output $p_\theta(y \mid x)$, and let labels be sorted so
$p_\theta(y_{(1)} \mid x) \ge \cdots \ge p_\theta(y_{(K)} \mid x)$.

### 1. Indirect + conformal
Train with cross-entropy, optionally apply post-hoc temperature scaling, then
form exact randomized split-conformal sets with score

$$
S_\theta(x,y)=p_\theta(y \mid x).
$$

For calibration scores $S_i^\theta=S_\theta(X_i,Y_i)$, the conformal p-value is

$$
p(x,y)=\frac{\sum_i \mathbf{1}\{S_i^\theta < S_\theta(x,y)\} + U\bigl(1+\sum_i \mathbf{1}\{S_i^\theta=S_\theta(x,y)\}\bigr)}{n+1},
$$

and the set is

$$
C^\alpha(x)=\{y : p(x,y)>\alpha\}.
$$

### 2. Direct + conformal
Train the score function end-to-end using the pure soft set-size surrogate
$\tilde L_{\text{size}}$, with conformal calibration inside the training
objective. Final evaluation still uses the exact conformal set

$$
C^\alpha(x)=\{y : p(x,y)>\alpha\}.
$$

### 3. Indirect + greedy cumulative mass, fixed $\tau$
Train indirectly, then build the set by accumulating top-ranked label mass
until $\tau = 1 - \alpha$:

$$
L_\tau(x)=\min\{\ell :
\sum_{j=1}^{\ell} p_\theta(y_{(j)} \mid x)\ge \tau \},
\qquad
C_{\theta,\tau}^{\text{mass}}(x)=\{y_{(1)},\dots,y_{(L_\tau(x))}\}.
$$

### 4. Indirect + greedy cumulative mass, calibrated $\hat{\tau}$
Train indirectly, then choose the smallest $\hat{\tau}$ on a calibration set
such that empirical coverage reaches the target:

$$\hat{\tau} = \inf\{\tau : \frac{1}{n}\sum_{i=1}^n \mathbf{1}\{Y_i \in C_{\theta,\tau}^{\text{mass}}(X_i)\} \ge 1 - \alpha \}.$$

The final set is

$$
C_{\theta,\hat{\tau}}^{\text{mass}}(x).
$$

## Datasets

- Toy fixed-point multiclass problem with an MLP over coordinates
- CIFAR-10 with ResNet-18
- CIFAR-100 hierarchy:
  - fine labels: 100 classes
  - coarse labels: 20 classes

## Run

Run the toy experiment:

```bash
python scripts/run_experiment.py --config configs/experiment/toy_v1.yaml
```

Run the CIFAR-10 experiment:

```bash
python scripts/run_cifar10_experiment.py \
  --config configs/experiment/cifar10_resnet18_v1.yaml
```

Run the CIFAR-100 hierarchy experiment:

```bash
python scripts/run_cifar100_hierarchy_experiment.py
```

Run only the CIFAR-100 fine or coarse experiment:

```bash
python scripts/run_cifar10_experiment.py \
  --config configs/experiment/cifar100_fine_resnet18_v1.yaml

python scripts/run_cifar10_experiment.py \
  --config configs/experiment/cifar100_coarse_resnet18_v1.yaml
```

Evaluate a saved toy checkpoint:

```bash
python scripts/evaluate_conformal.py \
  --config configs/experiment/toy_v1.yaml \
  --checkpoint outputs/toy_v1/indirect/seed_0/checkpoint.pt \
  --method indirect
```

## Outputs

Check `reports` for CIFAR-100 hierarchy experiment results. 

Each run writes a directory under:

```text
outputs/<experiment_name>/<method>/seed_<seed>/
```

Typical artifacts:

- `metrics.json`
- `train_history.json`
- `posthoc_calibration.json`
- `checkpoint.pt`
- split metadata (`splits.json` or `split_indices.json`)

## Notes

- The code uses `alpha` as **miscoverage**, so target coverage is \(1-\alpha\).
- Indirect-family methods may use post-hoc temperature scaling.
- Direct training does not use temperature scaling.
- Final conformal evaluation always uses an untouched outer calibration split.
