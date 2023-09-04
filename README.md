# Sync-NeRF : <br> Generalizing Dynamic NeRFs to Unsynchronized Videos

[![arXiv](https://img.shields.io/badge/arXiv-2110.02711-006600)]() 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://seoha-kim.github.io/sync-nerf/)
[![dataset](https://img.shields.io/badge/dataset-00A98F)](https://yonsei-my.sharepoint.com/:f:/g/personal/yj_uh_o365_yonsei_ac_kr/EshaQEg8FIZIqlU-mU8npikBIl8Rwk5Dvb6X6HvuFeU0_Q?e=GLdtqF/)

Official repository for the paper: <b>Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos</b><br>
Sync-NeRF enables dynamic NeRFs to successfully represent the dynamic scnene from time-unsynchroznied videos.

## Setup
We provide an integrated requirements file for K-Planes and MixVoxels.
>> pip install -r requirements.txt

## MixVoxels

## K-Planes
K-Planes offers two versions of config: hybrid and explicit. We provide example configs for the Unsynchronized Plenoptic Video Dataset and Unsynchronized Dynamic Blender Dataset. You can train the model using the following command:
>> PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py


After completing model training, you can perform evaluation using the --validate-only or --rendering-only flags. We also propose a method for optimizing time offsets during test time. You can execute this test-time optimization using the following command:
>> PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/logfolder --valid-only --test-optim