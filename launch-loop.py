import json
import os, sys
import pytorch_lightning as pl
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
    ProgressCallback,
)
from threestudio.systems.base import BaseSystem
from threestudio.utils.config import ExperimentConfig, load_config
from datetime import datetime

def main(args, extras) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    devices = -1
    if len(env_gpus) > 0:
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    pl.seed_everything(cfg.seed, workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    
    
    def set_system_status(system: BaseSystem, ckpt_path):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])


    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()
    # if not os.path.exists( cfg.trial_dir+"/gaussiansplatting"):
    #     shutil.copytree("./gaussiansplatting", cfg.trial_dir+"/gaussiansplatting")
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    # Load scenes from JSON
    with open(args.prompts_file, "r") as f:
        scenes = json.load(f)["scans"]

    # Loop through scenes and their objects
    for scene in scenes:
        scene_scan = scene["scan"]
        scene_style = scene["objstyle"]
        scene_objects = list(scene["objects"].values())[:-1]  # Exclude "floor"
        for idx, obj in enumerate(scene_objects, start=1):
            print(f"Running for scan: {scene_scan}, object: {obj}, {scene_style} style")

            # Update object prompt and *important* epoch/step
            system.cfg.prompt_processor.prompt = obj + f", {scene_style} style"
            system.set_resume_status(0, 0)

            # Create trial directory 
            # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            trial_leaf_dir_name = f"{idx}_{obj.replace(' ', '_')}"
            trial_dir = os.path.join("msnd", scene_scan, trial_leaf_dir_name)
            cfg.trial_dir = trial_dir
            cfg.name = f"{scene_scan}/{trial_leaf_dir_name}"
            cfg.trial_name = f"{scene_scan}/{trial_leaf_dir_name}"
            cfg.tag = f"{scene_scan}/{trial_leaf_dir_name}"

            # Set save directory
            system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
            print("system._save_dir", system._save_dir)

            if args.train:
                trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
                trainer.test(system, datamodule=dm)
                if args.gradio:
                    # also export assets if in gradio mode
                    trainer.predict(system, datamodule=dm)
            elif args.validate:
                # manually set epoch and global_step as they cannot be automatically resumed
                set_system_status(system, cfg.resume)
                trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
            elif args.test:
                # manually set epoch and global_step as they cannot be automatically resumed
                set_system_status(system, cfg.resume)
                trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
            elif args.export:
                set_system_status(system, cfg.resume)
                trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)
                

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used.")
    parser.add_argument("--prompts_file", required=False,
                        default="/home/ubuntu/datasets/FRONT/relationships_anyscene.json",
                        help="path to JSON file containing scenes")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument("--verbose", action="store_true", help="set logging level to DEBUG")
    parser.add_argument("--gradio", action="store_true", help="if true, run in gradio mode")
    
    args, extras = parser.parse_known_args()
    main(args, extras)
