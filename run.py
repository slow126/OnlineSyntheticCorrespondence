import os
import dotenv
from pytorch_lightning.cli import LightningCLI
import torch



if __name__ == '__main__':
    dotenv.load_dotenv('.env')

    from src import conf_resolvers  # registers custom resolvers

    torch.set_float32_matmul_precision('medium')

    cli = LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={'overwrite': True},
    )
