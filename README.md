# ECE 285: Deep Generative Models

My coursework for ECE 285, Deep Generative Models. Each homework has its own folder with its own code, results and README.

| Folder | Homework | Topic |
| --- | --- | --- |
| [HW1_VAE](HW1_VAE/) | Homework 1 | Convolutional variational autoencoder (VAE) that generates 128×128 chest X-rays, evaluated with FID and Inception Score |
| [HW2_GAN](HW2_GAN/) | Homework 2 | DCGAN that generates 64×64 paintings from the WikiArt dataset, evaluated with FID and Inception Score |

Every homework runs from inside its own folder, for example:

```bash
cd HW1_VAE
pip install -r requirements.txt
python run_all.py
```
