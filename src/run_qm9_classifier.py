# Rdkit import should be first, do not move it
from rdkit import Chem

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

import utils as utils
import datasets.qm9_dataset as qm9_dataset
from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from classifier.qm9_classifier_discrete import Qm9ClassifierDiscrete

import os

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.classifier.name, 'project': 'graph_ddm_regressor', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(dataset_config)
    assert dataset_config["name"] == 'qm9'
    assert cfg.classifier.type == 'discrete'
    datamodule = qm9_dataset.QM9DataModule(cfg, regressor=True)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    train_smiles = None

    if cfg.classifier.extra_features is not None:
        extra_features = ExtraFeatures(cfg.classifier.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    
    num_classes = cfg.classifier.num_classes
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': num_classes if cfg.classifier.guidance_target == 'both' else 1}
    # dataset_infos.input_dims['y'] = 0
    print(dataset_infos.input_dims)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    # model = Qm9ClassifierDiscrete(cfg=cfg, **model_kwargs)
    current_path = os.path.dirname(os.path.realpath(__file__))
    clf_path = os.path.join(current_path, cfg.classifier.trained_classifier_path)
    
    if(os.path.exists(clf_path) and (cfg.classifier.load_classifier == True)):
        model = Qm9ClassifierDiscrete.load_from_checkpoint(clf_path)
        print("Loaded Classifier Successfully")
        callbacks = []
        if cfg.classifier.save_model:
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.classifier.name}",
                                                filename='{epoch}',
                                                monitor='val/epoch_ce',
                                                save_last=True,
                                                save_top_k=-1,    # was 5
                                                mode='min',
                                                every_n_epochs=1)
            print("Checkpoints will be logged to", checkpoint_callback.dirpath)
            callbacks.append(checkpoint_callback)

        name = cfg.classifier.name
        if name == 'debug':
            print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
        trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                        accelerator='gpu' if len(cfg.general.gpus) > 0 and torch.cuda.is_available() else 'cpu',
                        devices=1 if len(cfg.general.gpus) > 0 and torch.cuda.is_available() else None,
                        limit_train_batches=20,     # TODO: remove
                        limit_val_batches=20 if name == 'test' else None,
                        limit_test_batches=20 if name == 'test' else None,
                        max_epochs=cfg.classifier.epochs,
                        check_val_every_n_epoch=cfg.classifier.check_val_every_n_epochs,
                        fast_dev_run=cfg.classifier.name == 'debug',
                        enable_progress_bar=False,
                        callbacks=callbacks)
        trainer.test(model, datamodule=datamodule)
    else:
        model = Qm9ClassifierDiscrete(cfg=cfg, **model_kwargs)
        callbacks = []
        if cfg.classifier.save_model:
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.classifier.name}",
                                                filename='{epoch}',
                                                monitor='val/epoch_ce',
                                                save_last=True,
                                                save_top_k=-1,    # was 5
                                                mode='min',
                                                every_n_epochs=1)
            print("Checkpoints will be logged to", checkpoint_callback.dirpath)
            callbacks.append(checkpoint_callback)

        name = cfg.classifier.name
        if name == 'debug':
            print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
        trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                        accelerator='gpu' if len(cfg.general.gpus) > 0 and torch.cuda.is_available() else 'cpu',
                        devices=1 if len(cfg.general.gpus) > 0 and torch.cuda.is_available() else None,
                        limit_train_batches=20,     # TODO: remove
                        limit_val_batches=20 if name == 'test' else None,
                        limit_test_batches=20 if name == 'test' else None,
                        max_epochs=cfg.classifier.epochs,
                        check_val_every_n_epoch=cfg.classifier.check_val_every_n_epochs,
                        fast_dev_run=cfg.classifier.name == 'debug',
                        enable_progress_bar=False,
                        callbacks=callbacks)

        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.classifier.resume)


if __name__ == '__main__':
    main()