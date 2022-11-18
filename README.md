# Neural Architecture for Online Ensemble Continual Learning

## Results reproduction

This project is developing under Python 3.8. Firstly you have to install project requirements:

```shell
pip install -r requirements.txt
```

Then you have to train the feature extractor for MNIST dataset. You can do this by running the notebooks/model/1-Omniglot-autoencoder.ipynb notebook. Model should be stored in storage/models/ensemble_omniglot_autoencoder/encoder.ckpt

To use the feature extractor for CIFAR-10 and CIFAR-100 datasets, please download it from https://github.com/yaox12/BYOL-PyTorch
The downloaded model should be stored in storage/models/resnet50_byol/resnet50_byol_imagenet2012.pth.tar

We strongly recommend using GPU for experiment reproduction.

To reproduce the experiment results for DE&E method run the following command from the main project directory:

```shell
PYTHONPATH=. python ./scripts/experiments/run_ensemble_e2e_experiment.py
```

To reproduce the experiment results for E&E method run the following command from the main project directory:

```shell
PYTHONPATH=. python ./scripts/experiments/run_ensemble_reproduction_experiment.py
```

To reproduce the experiment results for baseline methods run the following command from the main project directory:

```shell
PYTHONPATH=. python ./scripts/experiments/run_avalanche_experiment.py
```

Configurations for all experiments are stored in the mentioned scripts. So to 
change configuration of some experiments open the appropriate file (e.g. run_ensemble_e2e_experiment)
 and change the dictionary with configuration for the selected run.