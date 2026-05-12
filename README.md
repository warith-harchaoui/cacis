# CACIS: Cost-Aware Classification with Informative Sinkhorn




Classifiers deployed in industry trigger downstream actions whose consequences are rarely uniform across error types. Cross-entropy, the de facto training objective, is decision-agnostic: it treats every off-diagonal mistake identically and therefore aligns poorly with the actual quantity practitioners care about — the expected cost incurred by the model's decision.

We introduce **CACIS** (*Cost-Aware Classification using Informative Sinkhorn*), a differentiable loss that embeds a task-specific cost matrix into the geometry of the predictive distribution. CACIS uses entropy-regularized Optimal Transport to induce a non-isotropic metric on the label simplex and computes the model-implied distribution through a numerically stable Frank–Wolfe inner loop that does not require unrolling Sinkhorn iterations.

We illustrate the framework on two complementary examples that span the spectrum of ground costs. In **fraud detection** (*IEEE-CIS / Vesta*), the cost matrix is instance-dependent and monetary: each transaction induces its own cost via a value model parameterized by transaction amount, chargeback multiplier, and false-decline friction.

In **image recognition** (*ImageNet-1k with a torchvision ResNet trained from scratch*), the cost matrix is class-shared and semantic: the cost of confusing two classes is the cosine distance between the FastText embeddings of their names, so confusing *tiger* with *leopard* is penalized far less than confusing *tiger* with *minivan*.

The same loss, optimizer, and gradient strategy are used in both regimes; only the cost matrix changes. **CACIS** reduces realized regret relative to cross-entropy and weighted cross-entropy on the fraud benchmark while maintaining calibration, and produces semantically gentler errors on ImageNet at unchanged top-1 accuracy.The result is a single, decoupled recipe for cost-sensitive learning: specify the geometry once via a cost matrix, train as usual.


---

## 📍 Table of Contents

- [Why CACIS?](#-why-cacis)
- [Two illustrative examples](#-two-illustrative-examples)
- [Install](#-install)
- [Quick start](#-quick-start)
- [The four cost-aware losses](#-the-four-cost-aware-losses)
- [Example 1 — Fraud detection (IEEE-CIS / Vesta)](#-example-1--fraud-detection-ieee-cis--vesta)
- [Example 2 — ImageNet with FastText semantic cost](#-example-2--imagenet-with-fasttext-semantic-cost)
- [Inference](#-inference)
- [Local smoke test (before any cloud spend)](#-local-smoke-test-before-any-cloud-spend)
- [Launching on Vast.ai](#-launching-on-vastai)
- [Figure style: Montserrat + shared palette](#-figure-style-montserrat--shared-palette)
- [Repository layout](#-repository-layout)
- [Tests](#-tests)
- [Docs](#-docs)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Why CACIS?

Industrial classifiers don't return labels — they trigger **actions** whose consequences are almost never uniform.
Cross-entropy is decision-agnostic: every off-diagonal error contributes the same gradient.
**CACIS** replaces the isotropic Shannon/KL geometry of cross-entropy with the non-isotropic geometry
induced by *entropic optimal transport under a user-supplied cost matrix* $\mathbf C$.

- $\mathbf C_{i,j}$ = cost of predicting $j$ when truth is $i$ (zero on diagonal, non-negative off-diagonal)
- $\mathbf C$ can be **global** $(K,K)$ shared across the batch — or **instance-dependent** $(B,K,K)$
- The training loss "sees" the cost geometry directly. No threshold heuristics, no manual reweighting.

Read [`docs/math.md`](docs/math.md) for the derivation; [`docs/latex/`](docs/latex/) holds the KDD paper draft.

---

## 🧭 Two illustrative examples

The framework is **one recipe** — only the cost matrix changes:

| Axis | Example 1 — Fraud detection | Example 2 — ImageNet |
|------|----------------------------|----------------------|
| Cost varies with | each instance $\vec x$ (transaction amount) | each class pair $(i,j)$ |
| $K$ | 2 (approve / decline) | 1000 (WordNet) |
| $\mathbf C$ source | parametric business value model | cosine distance between FastText embeddings of class names |
| Units | dollars | dimensionless |
| Data | 0.5 M tabular transactions | 1.28 M images |
| Model | MLP / linear | torchvision ResNet from scratch |
| Compute | a single CPU | one RTX 4090 24 GB on Vast.ai |
| Where in repo | [`examples/fraud_detection.py`](examples/fraud_detection.py) | [`examples/imagenet/`](examples/imagenet/) |

The same `cost_aware_losses` package powers both. **Practitioners change $\mathbf C$, not the loss.**

---

## 🛠️ Install

```bash
git clone https://github.com/warith-harchaoui/cost_aware_classification.git cacis
cd cacis

# Recommended: conda (matches CI)
conda create -y -n env4cacis python=3.10
conda activate env4cacis
pip install -r requirements.txt
pip install -e .

# One-time: figure font (Montserrat) used by every plot
bash scripts/download_fonts.sh
```

To exercise the ImageNet sub-pipeline locally you'll also want `torchvision`, `kaggle`,
and a few small extras — they live in [`examples/imagenet/requirements.txt`](examples/imagenet/requirements.txt).

---

## ⚡ Quick start

**1. Validate the entire pipeline locally in ~30 s** (no GPU, no AWS, no Kaggle):

```bash
python -m examples.imagenet.smoke_test --loss all
```

The smoke test downloads **Imagenette** (10-class ImageNet subset from fast.ai, ~95 MB, cached at `~/.cache/cacis/`),
trains 2 epochs × 5 train + 2 val batches for **each of the four losses**, regenerates the curves from the
on-disk `metrics.jsonl`, runs `inference.py` in all three modes, and checks every artifact landed.
Exit code 0 = green, non-zero = something needs fixing — safe to wire into CI.

**2. Fraud benchmark on your laptop:**

```bash
python -m examples.fraud_detection --loss all --epochs 15 --run-id business_impact
```

**3. ImageNet on Vast.ai (~$30-50 for all 4 losses):** see [Launching on Vast.ai](#-launching-on-vastai).

---

## 📋 The four cost-aware losses

All inherit from [`cost_aware_losses.base.CostAwareLoss`](cost_aware_losses/base.py).
Same signature, drop-in for `nn.CrossEntropyLoss`:

```python
from cost_aware_losses import SinkhornEnvelopeLoss
loss_fn = SinkhornEnvelopeLoss()           # default ε from cost-matrix off-diagonal mean
loss = loss_fn(logits, targets, C=C)        # C: (K,K) shared, or (B,K,K) per-example
loss.backward()
```

| Loss | Gradient | Memory | Stability | Recommended for |
|------|----------|--------|-----------|-----------------|
| [`SinkhornFenchelYoungLoss`](cost_aware_losses/sinkhorn_fenchel_young.py) | Fenchel-Young + Frank-Wolfe inner solver | low | high | research, theory |
| [`SinkhornEnvelopeLoss`](cost_aware_losses/sinkhorn_envelope.py) | envelope (no backprop through Sinkhorn iters) | low | high | **default — production** ⭐ |
| [`SinkhornFullAutodiffLoss`](cost_aware_losses/sinkhorn_autodiff.py) | full autodiff through Sinkhorn iters | high | medium | research comparison |
| [`SinkhornPOTLoss`](cost_aware_losses/sinkhorn_pot.py) | envelope, [POT](https://pythonot.github.io/) backend | low | high | when you want POT's solver explicitly |

### Epsilon (ε) tuning

The entropic regularizer ε controls the smoothness of the transport plan. **By default it's data-adaptive:**

- `offdiag_mean` *(default)* — ε = mean of off-diagonal $\mathbf C$ entries × `epsilon_scale`
- `offdiag_median` — robust to outlier costs
- `offdiag_max` — most conservative
- `constant` — supply `epsilon=<float>` for controlled experiments

`epsilon_scale` (default 1.0) is a multiplier; tighten with `0.5`, loosen with `2.0`.

Optional **exponential-decay schedule** (`--epsilon-schedule exponential_decay`) starts at `10×` ε
for stable early training and decays to `0.1×` ε for sharp final decisions.

See [`docs/math.md`](docs/math.md) for the formal derivation of ε ↔ POT's `reg`.

---

## 💰 Example 1 — Fraud detection (IEEE-CIS / Vesta)

**Cost matrix is per-example and monetary.** From the value model in
[`docs/fraud_business_and_cost_matrix.md`](docs/fraud_business_and_cost_matrix.md):

$$
\mathbf C_i(\vec x_i) =
\begin{pmatrix}
0 & \rho_{\mathrm{FD}}\,M_i\\
\lambda_{\mathrm{cb}}\,M_i + F_{\mathrm{cb}} & 0
\end{pmatrix}
$$

where $M_i$ is the transaction amount, $\rho_{\mathrm{FD}}$ the false-decline friction (default 0.10),
$\lambda_{\mathrm{cb}}$ the chargeback multiplier (default 1.5), $F_{\mathrm{cb}}$ the fixed dispute fee (default \$15).

```bash
# Train all four losses + the two baselines, 15 epochs each
python -m examples.fraud_detection --loss all --epochs 15 --run-id business_impact

# Or one loss at a time
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
    --epsilon-mode offdiag_median --run-id env_median
```

Outputs land at `fraud_output/<run-id>/<loss>/`: per-epoch metrics CSV, PR curves, regret trajectories,
and `checkpoint_best.pt` selected by validation `expected_opt_regret`.

**Metrics that actually matter** (every figure is tagged with `↑ higher is better` or `↓ lower is better`):
- **Realized regret** (↓) — actual money lost following the model's decisions on holdout
- **Expected optimal regret** (↓) — theoretical floor under perfect calibration
- **Naive baseline** — better of "approve all" and "decline all" (the model has to beat this to be worth deploying)
- **PR-AUC** (↑) — sanity metric for class imbalance

The dataset is the [IEEE-CIS Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection) (Vesta).
Download once:

```bash
mkdir ieee-fraud-detection && cd ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip && unzip -q ieee-fraud-detection.zip
```

---

## 🖼️ Example 2 — ImageNet with FastText semantic cost

**Cost matrix is class-shared and semantic.** For each off-diagonal pair $(i,j)$:

$$
\mathbf C^{\mathrm{sem}}_{ij} = 1 - \cos\bigl(\mathrm{emb}(i), \mathrm{emb}(j)\bigr), \qquad \mathbf C^{\mathrm{sem}}_{ii}=0
$$

where $\mathrm{emb}(\cdot)$ is the [FastText](https://fasttext.cc/) Common Crawl English model
applied to each class's WordNet name (compound names tokenized and averaged; subword fallback for OOV tokens).
A confusion between *tiger* and *leopard* incurs ~0.07; *tiger* vs *minivan* ~1.05.

### Pipeline

1. **Build the cost matrix once** (4 MB output, runs locally — see [`scripts/build_cost_matrix.sh`](scripts/build_cost_matrix.sh)):
   ```bash
   bash scripts/build_cost_matrix.sh --fasttext /path/to/cc.en.300.bin
   # or auto-download FastText (~4.5 GB) to ~/.cache/cacis/
   bash scripts/build_cost_matrix.sh --download
   ```
   Uses the 1000-class WordNet mapping baked in at [`assets/imagenet/LOC_synset_mapping.txt`](assets/imagenet/LOC_synset_mapping.txt) — no Kaggle download needed for this step.

2. **Train on Vast.ai** (1× RTX 4090 24 GB, ~$30-50 for all four losses sequential):
   ```bash
   python -m examples.imagenet.cloud.launch_vast --config config/cloud-vast.yaml
   ```
   See the next section for full setup.

### Loss recommendation at K=1000

- ✅ **`sinkhorn_envelope`** — fits 24 GB at batch 32; the default.
- ✅ `cross_entropy` — baseline for regret-vs-CE comparison.
- ⚠️ `sinkhorn_autodiff` — memory-sensitive; reduce `sinkhorn_max_iter` to 10.
- ⚠️ `sinkhorn_pot` — slow at K=1000 because POT solves per example.

The adaptive ε statistic is **precomputed once at startup** from the shared cost matrix and passed as `epsilon_mode='constant'` to the loss — avoiding a per-batch reduction over a $(B,1000^2)$ mask.

---

## 🔮 Inference

[`examples/imagenet/inference.py`](examples/imagenet/inference.py) loads a checkpoint and supports three modes. When a cost matrix is supplied it reports **both** the standard `argmax` prediction *and* the **cost-optimal action**:

$$
\hat a(\vec p) = \arg\min_a\; \langle \vec p, \mathbf C_{:,a}\rangle
$$

```bash
# Single image: top-5 + cost-optimal label
python -m examples.imagenet.inference \
    --checkpoint imagenet_output/<run>/checkpoint_best.pt \
    --cost-matrix cost_matrix.pt \
    --image /path/to/photo.jpg --topk 5

# Directory → predictions.csv
python -m examples.imagenet.inference \
    --checkpoint .../checkpoint_best.pt --cost-matrix cost_matrix.pt \
    --input-dir /path/to/photos/ --output predictions.csv

# ImageFolder val set: Top-1, Top-5, realized regret under BOTH decision rules
python -m examples.imagenet.inference \
    --checkpoint .../checkpoint_best.pt --cost-matrix cost_matrix.pt \
    --val-dir /data/imagenet/val
```

The full-validation mode doubles as a regret-vs-argmax ablation.

---

## 🧪 Local smoke test (before any cloud spend)

**Always run this first.** Validates the *entire* pipeline on real images in ~30 s — no GPU required.

```bash
python -m examples.imagenet.smoke_test                # default loss (sinkhorn_envelope)
python -m examples.imagenet.smoke_test --loss all     # all four
python -m examples.imagenet.smoke_test --keep         # preserve the workdir for inspection
```

The smoke test:

1. Downloads **Imagenette** (10-class ImageNet subset from fast.ai, ~95 MB, cached at `~/.cache/cacis/`) on first run.
2. Samples 8 train + 4 val + 4 test real JPEGs per class into a temporary `ImageFolder` layout.
3. Builds a synthetic random (10, 10) cost matrix (smoke test validates the *pipeline*, not the embeddings).
4. Trains 2 epochs × 5 train + 2 val batches on CPU/MPS with the requested loss(es).
5. **Re-renders the curves from the on-disk `metrics.jsonl`** via `examples.imagenet.plots` — proving each figure is reproducible from data alone.
6. Runs `inference.py` in all three modes against the synthetic test split.
7. Verifies every required artifact landed (`args.json`, `metrics.jsonl`, `metrics.csv`, both checkpoints, three curve PNGs, `predictions.csv`).

Exit code 0 on success; non-zero on any failure (CI-safe).

### Regenerating curves later

Every figure is re-creatable from `metrics.jsonl` alone:

```bash
python -m examples.imagenet.plots --run-dir imagenet_output/<run-id>/
```

Rewrites `curve_loss.png`, `curve_accuracy.png`, `curve_regret.png`, and `metrics.csv` from the JSONL.

---

## ☁️ Launching on Vast.ai

One Vast.ai instance (RTX 4090 24 GB, ~$0.30-0.45/hr), four losses sequential, ImageNet pulled from Kaggle inside the box, outputs rclone'd to Backblaze B2, self-destroys when done.

**Estimated total: ~$30-50 for all four losses** at default settings (45 epochs, batch 32). Full ResNet-50 × 90 epochs comes in around $120-200.

The full beginner walkthrough lives in [**`config/README.md`**](config/README.md) (Vast.ai signup → Kaggle → B2 → Docker Hub → cost matrix → first launch). Short version:

```bash
# Once: install vastai CLI, start a local docker daemon (Colima), copy the template
pip install vastai PyYAML
vastai set api-key <YOUR-KEY>

brew install colima docker-buildx   # macOS daemon for building the image
colima start --cpu 4 --memory 8 --disk 60

cp config/cloud-vast.yaml.example config/cloud-vast.yaml
# edit cloud-vast.yaml — credentials accept env-var NAMES or literal values

# Dry-run (no Vast.ai calls)
python -m examples.imagenet.cloud.launch_vast --config config/cloud-vast.yaml --dry-run

# Actually launch
python -m examples.imagenet.cloud.launch_vast --config config/cloud-vast.yaml
```

The launcher prints the instance id and a `vastai logs <id> --follow` command. Outputs appear under `<rclone_remote>/<run_id>/<run_id>-<loss>/` as each loss finishes, and the instance self-terminates when training completes.

For all four losses in **parallel** (¼ the wall clock, same total cost), launch the script four times with `losses: [<single_loss>]` in four separate config files.

---

## 🎨 Figure style: Montserrat + shared palette

Every figure in the repo — fraud + ImageNet, training curves + PR curves — uses **Montserrat-Regular** and a single Apple-style palette mirroring [harchaoui.org/warith/colors](https://harchaoui.org/warith/colors). The setup is centralized in `examples/utils.py::setup_plot_style()` and applied at import time.

The font is **not** tracked in git (binaries don't belong there). Fetch it once:

```bash
bash scripts/download_fonts.sh           # idempotent; skips if already present
bash scripts/download_fonts.sh --force   # re-download
```

If the file is missing at runtime, plots fall back to default sans-serif with a single warning — nothing crashes. The Docker image builds the font into itself, so cloud runs are unaffected.

Every figure title carries an explicit direction indicator: `↑ higher is better` or `↓ lower is better`.

---

## 🗂️ Repository layout

```
cost_aware_losses/                Core library — 4 losses + shared base class
├── base.py                       CostAwareLoss ABC, cost-matrix handling, ε computation
├── sinkhorn_fenchel_young.py     Fenchel-Young loss + Frank-Wolfe inner solver
├── sinkhorn_envelope.py          Custom Sinkhorn with envelope gradient (recommended)
├── sinkhorn_autodiff.py          Custom Sinkhorn with full autodiff
└── sinkhorn_pot.py               POT library backend with envelope gradient

examples/
├── fraud_detection.py            IEEE-CIS / Vesta benchmark runner
├── tabular_models.py             Linear / MLP backbones for fraud
├── utils.py                      Plot style (Montserrat, palette), TrainingState, plotting helpers
├── harvest_results.py            Collect summary.csv across fraud runs → LaTeX table
└── imagenet/
    ├── cost_matrix.py            FastText cost-matrix builder (CLI)
    ├── classnames.py             ImageNet class-name loader
    ├── data.py                   DDP-aware ImageNet loaders
    ├── model.py                  torchvision ResNet factory (from scratch)
    ├── train.py                  Single-GPU + AMP training, resumable, with --quick smoke mode
    ├── inference.py              argmax + cost-optimal prediction, three CLI modes
    ├── plots.py                  Curves from metrics.jsonl, re-runnable post-hoc
    ├── smoke_test.py             ~30 s end-to-end pipeline validation
    ├── Dockerfile                Image pushed to Docker Hub for Vast.ai
    ├── requirements.txt          ImageNet-specific extras
    └── cloud/
        ├── config.py             Typed loader for cloud-vast.yaml
        ├── launch_vast.py        Searches cheapest offer, creates instance, ships env + bootstrap
        └── vast_bootstrap.sh     Runs in the container: Kaggle → train loop → rclone → self-destroy

config/
├── README.md                     Step-by-step Vast.ai setup (start here for cloud)
├── cloud-vast.yaml.example       Annotated template; copy to cloud-vast.yaml (gitignored)

scripts/
├── build_cost_matrix.sh          Wraps cost_matrix.py with auto-FastText-locate / --download
└── download_fonts.sh             Fetch Montserrat-Regular.ttf into assets/fonts/

assets/
├── imagenet/LOC_synset_mapping.txt   1000-class WordNet mapping, baked in
└── fonts/                            Montserrat-Regular.ttf (gitignored, fetched once)

docs/
├── math.md                       Mathematical foundations (ε ↔ POT reg, Sinkhorn, FY)
├── fraud_business_and_cost_matrix.md   Value model → regret matrix derivation
└── latex/                        KDD paper draft (anonymous version)

tests/
├── test_sinkhorn_consistency.py  Cross-implementation loss + gradient equivalence
└── test_sinkhorn_advanced.py     gradcheck, ε→0 limit, extreme costs, shift invariance
```

---

## 🔬 Tests

```bash
pip install -e .   # if not already done
pytest tests
```

The suite includes:

- **Consistency** — `SinkhornPOTLoss`, `SinkhornEnvelopeLoss`, `SinkhornFullAutodiffLoss` must produce matching loss values and gradients on the same inputs (with documented tolerances for full-autodiff vs envelope).
- **`gradcheck`** — `torch.autograd.gradcheck` with finite differences mathematically proves the analytical "gradient grafting" used in `SinkhornPOTLoss` / `SinkhornEnvelopeLoss`.
- **ε → 0 limit** — Sinkhorn cost converges to exact Earth Mover's Distance as ε shrinks.
- **Extreme costs** — no `NaN` / `Inf` outputs at $\mathbf C$ values up to $10^5$.
- **Cost-shift invariance** — adding a constant $k$ to $\mathbf C$ raises the loss by exactly $k$ and leaves gradients invariant.

Plus the end-to-end smoke test (`python -m examples.imagenet.smoke_test --loss all`) covers the full pipeline including the cloud-launcher's config loader and the inference CLI.

---

## 📚 Docs

- [**`docs/math.md`**](docs/math.md) — Mathematical foundations: entropic OT, Sinkhorn iterations, Fenchel-Young framework, ε ↔ POT.reg mapping, adaptive-ε heuristics.
- [**`docs/fraud_business_and_cost_matrix.md`**](docs/fraud_business_and_cost_matrix.md) — "Geometry of regret" derivation for IEEE-CIS: value matrix → cost matrix, choice of $\rho_{\mathrm{FD}}, \lambda_{\mathrm{cb}}, F_{\mathrm{cb}}$.
- [**`docs/latex/`**](docs/latex/) — KDD '26 ADS paper draft (anonymized for review).
- [**`config/README.md`**](config/README.md) — 10-step Vast.ai walkthrough from zero.

---

## ✍️ Citation

```bibtex
@inproceedings{harchaoui2026cacis,
  title     = {Cost-Aware Classification: Putting Geometry into Label Distributions
               for Fraud Detection and Image Recognition},
  author    = {Harchaoui, Warith and Pantanacce, Laurent},
  booktitle = {The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Paris, France}
}
```

---

## 📜 License

**Unlicense** — free and unencumbered software released into the public domain.
See [UNLICENSE](https://unlicense.org) for details.

## 🙏 Acknowledgments

- [Python Optimal Transport (POT)](https://pythonot.github.io/) — Cuturi-era Sinkhorn implementation we wrap.
- [fast.ai Imagenette](https://github.com/fastai/imagenette) — the 10-class ImageNet subset used by the smoke test.
- [Vesta / IEEE-CIS Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection) — the fraud benchmark dataset.
- [FastText](https://fasttext.cc/) — the embedding model behind the ImageNet semantic cost matrix.

And for the fruitful discussions that made this work possible:



- [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug/)
- [Edmond Jacoupeau](https://www.linkedin.com/in/edmond-jacoupeau/)
