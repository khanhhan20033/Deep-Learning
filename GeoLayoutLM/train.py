import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from lightning_modules.data_modules.vie_data_module import VIEDataModule
from lightning_modules.geolayoutlm_vie_module import GeoLayoutLMVIEModule
from utils import get_callbacks, get_config, get_loggers, get_plugins
def main():
    cfg = get_config()
    print(cfg)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Avoid memory fragmentation

    seed_everything(cfg.seed)

    callbacks = get_callbacks(cfg)
    plugins = get_plugins(cfg)
    loggers = get_loggers(cfg)

    # Initialize model
    pl_module = GeoLayoutLMVIEModule(cfg)

    for attempt in range(3):  # Retry up to 3 times with reduced batch size
        try:
            # Initialize data module with the current batch size
            data_module = VIEDataModule(cfg, pl_module.net.tokenizer)

            trainer = Trainer(
                accelerator=cfg.train.accelerator,
                gpus=torch.cuda.device_count(),
                max_epochs=cfg.train.max_epochs,
                gradient_clip_val=cfg.train.clip_gradient_value,
                gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
                callbacks=callbacks,
                plugins=plugins,
                sync_batchnorm=True,
                precision=16 if cfg.train.use_fp16 else 32,  # Use mixed precision if specified
                detect_anomaly=False,
                replace_sampler_ddp=False,
                move_metrics_to_cpu=False,
                progress_bar_refresh_rate=0,
                check_val_every_n_epoch=cfg.train.val_interval,
                logger=loggers,
                benchmark=cfg.cudnn_benchmark,
                deterministic=cfg.cudnn_deterministic,
                limit_val_batches=cfg.val.limit_val_batches,
                # Removed accumulate_grad_batches
            )

            trainer.fit(pl_module, datamodule=data_module)
            break  # Exit loop if training succeeds

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("CUDA memory cleared due to OOM error. Reducing batch size.")
            cfg.train.batch_size = max(1, cfg.train.batch_size // 2)  # Reduce batch size
            if attempt == 2:
                raise RuntimeError("Failed to fit model due to OOM errors after multiple attempts")

if __name__ == "__main__":
    main()
