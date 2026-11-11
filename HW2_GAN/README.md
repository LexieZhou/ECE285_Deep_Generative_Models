# Homework 2: Generating Paintings with a GAN

For this homework I implemented a DCGAN in PyTorch, trained it on the WikiArt painting dataset, generated new 64×64 paintings from random noise, and scored them with FID and Inception Score.

## What the homework covers

1. **Model.** A DCGAN (Radford et al., 2016): a transposed-conv generator and a strided-conv discriminator. Two optional stabilizers can be compared against it: spectral normalization in the discriminator and the hinge loss.
2. **Data.** The [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart): about 81k paintings in 27 style folders. The GAN is unconditional and ignores the styles.
3. **Training.** Training alternates discriminator and generator steps. Loss curves, D(x) / D(G(z)) curves, sample grids and a periodic quick FID are saved along the way.
4. **Evaluation.** FID and Inception Score are computed with PyTorch-Ignite as in its [GAN evaluation blog post](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

## Model

| | Generator | Discriminator |
| --- | --- | --- |
| Input → output | z ∈ ℝ¹⁰⁰ ~ N(0, I) → 3×64×64 image in [-1, 1] | 3×64×64 image → one real/fake logit |
| Layers | 4×4 transposed convs: z → 512×4² → 256×8² → 128×16² → 64×32² → 3×64² | 4×4 stride-2 convs: 3×64² → 64×32² → 128×16² → 256×8² → 512×4² → 1 |
| Normalization | BatchNorm (none on the output) | BatchNorm (none on the input), or spectral norm with `--d_norm spectral` |
| Activations | ReLU, tanh output | LeakyReLU(0.2), no sigmoid (the losses take logits) |
| Parameters | 3.58M | 2.77M |

Weights are initialized from N(0, 0.02), as in DCGAN. `--img_size` is not an option: the size comes from the `.npy` file, and any power of 2 works (128×128 adds one layer to each network).

**Losses.**
- `--loss bce` (the default) is the standard GAN loss. D minimizes −log D(x) − log(1 − D(G(z))). G minimizes the non-saturating loss −log D(G(z)) rather than log(1 − D(G(z))), which has almost no gradient early on, when D rejects every fake. `--real_label 0.9` adds one-sided label smoothing.
- `--loss hinge` is the hinge loss of SNGAN/BigGAN. D minimizes max(0, 1 − D(x)) + max(0, 1 + D(G(z))). G minimizes −D(G(z)).

## Data preparation

[prepare_data.py](prepare_data.py) reads every painting once, center-crops it to a square (keeping the shorter side whole) and resizes it to 64×64 with Lanczos filtering. All the images go into one uint8 array, `data/wikiart_64.npy` (about 1 GB). Decoding the original JPEGs, which are often 1000–2000 px, on every epoch would make data loading the bottleneck. Paintings are center-cropped rather than squashed, because squashing would distort the composition. During training, images are flipped horizontally at random.

## Setup (on the server)

```bash
cd HW2_GAN
pip install -r requirements.txt
```

The download needs a [Kaggle API token](https://www.kaggle.com/docs/api) in `~/.kaggle/kaggle.json`. The archive is about 30 GB, so check your free disk space first with `df -h .`. Don't pass `--unzip`: the zip and the extracted files would need about 60 GB at once, and `prepare_data.py` reads the images straight from the zip.

```bash
kaggle datasets download -d steubk/wikiart -p data          # writes data/wikiart.zip
python prepare_data.py --src data/wikiart.zip --size 64     # writes data/wikiart_64.npy and _meta.json
```

`data/` is git-ignored. Once the `.npy` file exists, you can delete `data/wikiart.zip`. If the disk the repo is on is too small, download to a larger one (`-p /path/to/big/disk`) and point `--src` there. If preprocessing runs out of RAM, lower `--workers`.

## Usage

Run every command from this folder (`HW2_GAN/`). On a remote server, run long jobs inside `tmux` (or with `nohup ... &`) so they survive a dropped SSH connection.

| Command | What it does |
| --- | --- |
| `python run_all.py --out_dir outputs/run1` | Trains, then runs FID/IS on `best.pt`. It takes the same arguments as `train.py`. |
| `python train.py --out_dir outputs/run1` | Trains only. Writes `config.json`, `train.log`, `history.csv`, `fid_history.csv`, `loss_curve.png`, a sample grid per epoch in `samples/`, `best.pt` (lowest quick FID), `G_epochNNN.pt` every 10 epochs and `last.pt`. |
| `python train.py --out_dir outputs/run1 --resume` | Continues from `last.pt` after the job was killed. Pass the same arguments again, with `--epochs` set to the total number of epochs you want. |
| `python fid_is.py --ckpt outputs/run1/best.pt` | FID and IS on 10k generated vs. 10k real images, over 3 seeds, plus a real-vs-real baseline. Writes `fid_is_best.json` and `generated_best.png`. |

Run `python train.py --help` to see every hyperparameter. Suggested runs to compare in the report:

```bash
python run_all.py --out_dir outputs/run1                                    # DCGAN baseline
python run_all.py --out_dir outputs/run2 --d_norm spectral --loss hinge     # spectral norm + hinge
```

## Default hyperparameters

| Setting | Value |
| --- | --- |
| Image size | 64 × 64 RGB, center-cropped, random horizontal flip |
| Latent dimension | 100, z ~ N(0, I) |
| Optimizer | Adam, lr 2e-4 for both G and D, β = (0.5, 0.999) |
| Batch size / epochs | 128 / 50 (about 630 iterations per epoch) |
| D steps per G step | 1 |
| Loss | non-saturating BCE |

## Evaluation

FID and IS follow the ignite blog. Images are mapped from [-1, 1] to [0, 1], resized to 299×299 with PIL bilinear, and fed to ignite's default extractor, Inception-v3's **1000-way softmax**. The one difference from the blog is that the blog passes tanh outputs in [-1, 1] straight to `ToPILImage`, which mangles negative values; here they are mapped to [0, 1] first.

FID on softmax probabilities is on a much smaller scale than the usual FID on 2048-d pool features: values are typically below 1, not in the tens. So it can't be compared with numbers from papers, only with other runs here and with the real-vs-real baseline. The baseline is the FID between two disjoint sets of real paintings of the same size, which is the score a perfect generator would get.

During training, a quick FID (2,000 vs. 2,000 images, one fixed seed) is computed every 5 epochs, and the best checkpoint is kept as `best.pt`. A GAN has no validation loss, and its losses say little about image quality, so FID is used to pick the checkpoint instead.

## Results

*To be filled in after training on the server.*

## Files

```
HW2_GAN/
├── prepare_data.py  resize every painting once into data/wikiart_<size>.npy
├── dataset.py       .npy dataset, flips, [-1, 1] scaling, data loader
├── model.py         DCGAN generator and discriminator, BCE and hinge losses
├── metrics.py       ignite FID / IS helpers shared by train.py and fid_is.py
├── train.py         training loop, curves, sample grids, quick FID, checkpoints, resume
├── fid_is.py        final FID / IS with a real-vs-real baseline
└── run_all.py       train.py followed by fid_is.py on best.pt
```

Checkpoints (`*.pt`) and the dataset are git-ignored.
