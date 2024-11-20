import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from classifier.qm9_classifier_discrete import Qm9ClassifierDiscrete

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    print(resume)
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    
    new_cfg = cfg.copy()
    for category in cfg:
        if(category in model.cfg):
            for arg in cfg[category]:
                if(arg in model.cfg[category]):
                    new_cfg[category][arg] = model.cfg[category][arg]
    new_cfg.general.test_only = resume
    new_cfg.general.name = name
    new_cfg = utils.update_config_with_new_keys(new_cfg, new_cfg)

    # cfg = model.cfg
    # cfg.general.test_only = resume
    # cfg.general.name = name
    # cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return new_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg
    
    for category in cfg:    
        if(category not in new_cfg):
            new_cfg = cfg        
        for arg in cfg[category]:
            if(arg not in new_cfg[category]):
                new_cfg[category] = cfg[category]
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model

def reconstruct_dataset(num_recon, diffusion_model, classifier):
    pass


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    device_ids = cfg.general.gpus
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(cfg.general.gpus[0]))
    print(device)

    if dataset_config["name"] in ['qm9']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
            # test_smiles = qm9_dataset.get_test_val_smiles(cfg=cfg, train_dataloader=datamodule.test_dataloader(),
            #                                             dataset_infos=dataset_infos, split_flag = 'test', evaluate_dataset=False)
            # val_smiles = qm9_dataset.get_test_val_smiles(cfg=cfg, train_dataloader=datamodule.val_dataloader(),
            #                                             dataset_infos=dataset_infos, split_flag = 'val', evaluate_dataset=False)
        else:
            raise ValueError("Dataset not implemented")
        

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features, num_classes = None)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)    # train_smiles is used when the training dataloader is actually the test_dataset
        print(cfg.dataset)
        print(cfg.dataset.remove_h)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, model = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, model = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])
    else:
        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
    utils.create_folders(cfg)
        
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    cfg.general.gpus = device_ids 
    use_gpu = len(cfg.general.gpus) > 0 and torch.cuda.is_available()

    '''initialize classifier'''
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}
    current_path = os.path.dirname(os.path.realpath(__file__))
    clf_path = os.path.join(current_path, 'checkpoints/classifier_qm9.ckpt')
    if(os.path.exists(clf_path)):
        print("Load classifier from checkpoint ")
        classifier = Qm9ClassifierDiscrete.load_from_checkpoint(clf_path)
    else:
        classifier = Qm9ClassifierDiscrete(cfg=cfg, **model_kwargs)

    '''Reconstruction'''
    from reconstruction.reconstruct import kkt_reconstruction
    from torch.utils.tensorboard import SummaryWriter   

    reconstructor = kkt_reconstruction(cfg, dataset_infos, model, classifier, datamodule, device = device)

    recon_samples = reconstructor.reconstruct()
    sampling_metrics.forward(recon_samples, cfg.general.name, 0, val_counter=-1, test=False,
                                          local_rank=0)

if __name__ == '__main__':
    main()
