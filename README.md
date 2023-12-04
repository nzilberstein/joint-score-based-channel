# Joint channel estimation and data detection in massive MIMO systems based on diffusion models

Joint Massive MIMO channel estimation and data detector based on score-based (diffusion using annealed Langevin dynamics) generative model

This repo contains the official implementation of the "Joint channel estimation and data detection in massive MIMO systems based on diffusion models" paper (arXiv.2205.05776, link: https://arxiv.org/abs/2311.10311) 

## Code 

After cloning the repository, in a clean Python environment run `pip install -r requirements.txt`.

### Data generation

We train using 3GPP model using the QuaDriGa generator. You will need Matlab to run "simulate_FD_MIMO_sector.m" (located in the `quadriga` folder)
If you do not have Matlab or you do not want to generate your own channels, you can download from https://drive.google.com/drive/folders/1uHxGHTyU4SXyHYXvPtjnekEKCVNfb24r?usp=sharing.
Train and validation data is in the same file.

Once downloaded, place these files in the `data` folder under the main directory.

### Pre-trained Models
A pre-trained diffusion model for 3GPP channels can be directly downloaded from https://drive.google.com/drive/folders/1uHxGHTyU4SXyHYXvPtjnekEKCVNfb24r?usp=sharing.

Once downloaded, place the model in `models` as `model_3gpp_64x32.pt`

## Training 
After downloading the example 3GPP data, a diffusion model can be trained by running:

```python train_score.py```

The model is trained for 80 epochs by default, and the last model weights will be automatically saved in the `model` folder under the appropriate structure. To train on other channel distributions, see the `--train` argument.

## Inference
To run channel estimation with the 3GPP data and the pretrained model run:

```python test_joint.py```

This will run channel estimation in the setting of paper with P = 30 and D = 50 of the paper; see the corresponding result in:

<embed src="https://github.com/nzilberstein/joint-score-based-channel/blob/main/figures/recon_SER_64x32.pdf" width="500" height="375">
<embed src="https://github.com/nzilberstein/joint-score-based-channel/blob/main/figures/recon_symbs_64x32.pdf width="500" height="375">

Running the above command will automaticall save results in the `results_seed%f/3GPP_numpilots%f_numsymbols%f.pt` folder. 

# Acknowledgement

This repo is largely based on the score-based-channels repo (in particular the training and testing w.r.t. the channel as well as the baseline methods); several functions are similar to them (link: https://github.com/utcsilab/score-based-channels/tree/main).
And also is based on the MIMO detector based on Langevin (link: https://github.com/nzilberstein/Langevin-MIMO-detector).

# Citations
Please include the following citation when using or referencing this codebase:

``` 
@article{zilberstein2023joint,
  title={Joint channel estimation and data detection in massive MIMO systems based on diffusion models},
  author={Zilberstein, Nicolas and Swami, Ananthram and Segarra, Santiago},
  journal={arXiv preprint arXiv:2311.10311},
  year={2023}
}
 ```