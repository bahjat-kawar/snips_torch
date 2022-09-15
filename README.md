# SNIPS: Solving Noisy Inverse Problems Stochastically

This repo contains the official implementation for the paper [SNIPS: Solving Noisy Inverse Problems Stochastically](http://arxiv.org/abs/2105.14951). 

by Bahjat Kawar, Gregory Vaksman, and Michael Elad, Computer Science Department, Technion.

## Running Experiments

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the snips environment.

```bash
conda env create -f environment.yml
```

The environment includes `cudatoolkit=11.0`. You may change that depending on your hardware.

### Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [-i IMAGE_FOLDER]
               [-n NUM_VARIATIONS] [-s SIGMA_0] [--degradation DEGRADATION]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  -n NUM_VARIATIONS, --num_variations NUM_VARIATIONS
                        Number of variations to produce
  -s SIGMA_0, --sigma_0 SIGMA_0
                        Noise std to add to observation
  --degradation DEGRADATION
                        Degradation: inp | deblur_uni | deblur_gauss | sr2 |
                        sr4 | cs4 | cs8 | cs16

```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   └── lsun # all LSUN files
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      └── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
├── image_samples # contains generated samples
│   └── <i>
│       ├── stochastic_variation.png # samples generated from checkpoint_x.pth, including original, degraded, mean, and std   
│       ├── results.pt # the pytorch tensor corresponding to stochastic_variation.png
│       └── y_0.pt # the pytorch tensor containing the input y of SNIPS
```

### Downloading data

You can download the aligned and cropped CelebA files from their official source [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The LSUN files can be downloaded using [this script](https://github.com/fyu/lsun). For our purposes, only the validation sets of LSUN bedroom and tower need to be downloaded.

### Running SNIPS

If we want to run SNIPS on CelebA for the problem of super resolution by 2, with added noise of standard deviation 0.1, and obtain 3 variations, we can run the following

```bash
python main.py -i celeba --config celeba.yml --doc celeba -n 3 --degradation sr2 --sigma_0 0.1
```

Samples will be saved in `<exp>/image_samples/celeba`.

The available degradations are: Inpainting (`inp`), Uniform deblurring (`deblur_uni`), Gaussian deblurring (`deblur_gauss`), Super resolution by 2 (`sr2`) or by 4 (`sr4`), Compressive sensing by 4 (`cs4`), 8 (`cs8`), or 16 (`cs16`). The sigma_0 can be any value from 0 to 1.

## Pretrained Checkpoints

Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing

These checkpoint files are provided as-is from the authors of [NCSNv2](https://github.com/ermongroup/ncsnv2). You can use the CelebA, LSUN-bedroom, and LSUN-tower datasets' pretrained checkpoints. We assume the `--exp` argument is set to `exp`.

## Acknowledgement

This repo is largely based on the [NCSNv2](https://github.com/ermongroup/ncsnv2) repo, and uses modified code from [this repo](https://github.com/alisaaalehi/convolution_as_multiplication) for implementing the blurring matrix.

## References

If you find the code/idea useful for your research, please consider citing

```bib
@article{kawar2021snips,
  title={{SNIPS}: Solving noisy inverse problems stochastically},
  author={Kawar, Bahjat and Vaksman, Gregory and Elad, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={21757--21769},
  year={2021}
}
```

