
## Training models with PyTorch Lightning

To train a new model, construct a config.yaml file at ```train/configs/experiment/```. 

You can launch a training job using the following command from the ```train/``` directory, where you can modify the config name and number of GPUs (```trainer.devices```):
```bash
cd train/
python run.py experiment=tk/owt_tk_gpts trainer.devices=1
```

By default, training will occur on OpenWebText data, downloaded from Hugging Face. Details for the dataset preparation (e.g., the cache directory if you wish to set this) are controlled at the yaml config: `train/configs/datamodules/openwebtext.yaml`.

