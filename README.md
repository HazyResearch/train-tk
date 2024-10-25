

## Training with ThunderKittens

This repo provides example code to train with the ThunderKittens attention kernels. 

Structure:

PyTorch Lightning
- ./train/ includes PyTorch Lightning training code, the README there describes how to get started
- ./tktrainer/ includes the corresponding PyTorch modules with the implementations
To set this up:
```bash
python setup.py install
```

Sample command, which will is controlled by the yaml config at: ```train/configs/experiments/tk```:
```bash
cd train/
python run.py experiment=tk/owt_tk_gpts trainer.devices=1
```


nanoGPT
- ./nano-train/ includes nanoGPT training code, the README there describes how to get started



