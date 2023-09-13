# train script
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py

# test optim(eval) script
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/logfolder --valid-only --test_optim