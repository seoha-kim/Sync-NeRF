# Sync-NeRF : Generalizing Dynamic NeRFs <br> to Unsynchronized Videos

[![arXiv](https://img.shields.io/badge/arXiv-2310.13356-006600)](https://arxiv.org/abs/2310.13356) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://seoha-kim.github.io/sync-nerf/)
[![dataset](https://img.shields.io/badge/dataset-00A98F)](https://drive.google.com/drive/folders/1wvLtucVrmFf7fj-kWr-HMk3boaI46cIX?usp=sharing)

Official repository for <a href="https://arxiv.org/abs/2310.13356">Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos</a><br>
enabling dynamic NeRFs to successfully reconstruct the scene from unsynchroznied dataset.
<p align="center" width="100;">
<img src="https://github.com/seoha-kim/Sync-NeRF/assets/46925617/616278bb-4bb5-41c5-8f57-12242c8403e0">
</p>
<br>

## Setup
We provide an integrated requirements file for Sync-MixVoxels and Sync-K-Planes.
```
pip install -r requirements.txt
```
<br>

You can download our Unsynchronized Dynamic Blender Dataset from this <a href="https://drive.google.com/drive/folders/1wvLtucVrmFf7fj-kWr-HMk3boaI46cIX?usp=sharing">link</a>
<br>
<br>

## Sync-MixVoxels
We provide example configs for the Unsynchronized Plenoptic Video Dataset and Unsynchronized Dynamic Blender Dataset. You can train the model using the following command:
```
python train.py --config path/to/config.txt
```
We also propose a method for optimizing time offsets during test time. You can execute this test-time optimization using the following command:
```
python train.py --config path/to/config.txt --test-optim
```

After completing model training, you can perform evaluation using the --render_only 1 flags.
```
python train.py --config path/to/config.txt --render_only 1 --ckpt path/to/checkpoint.pt
```

<br>

## Sync-K-Planes
K-Planes offers two versions of config: hybrid and explicit. We provide example configs for the Unsynchronized Plenoptic Video Dataset and Unsynchronized Dynamic Blender Dataset. You can train the model using the following command:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py
```

We also propose a method for optimizing time offsets during test time. You can execute this test-time optimization using the following command:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/logfolder --test_optim
```

After completing model training, you can perform evaluation using the --validate-only or --rendering-only flags.

## Citation
```latex
@article{Kim2024Sync,
  title={Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos},
  author={Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee, Gun Bang, Youngjung Uh},
  booktitle={AAAI},
  year={2024}}
```
