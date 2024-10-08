import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from IPython.display import display
import lightning as L
from lightning.pytorch import seed_everything
from simulation import r_middle_to_dv
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from multiprocessing import Pool



class ImageDataset(Dataset):
    """
    given data_dir, which as res_map and vol_map subdirs, creates a dataset
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir: Path = data_dir
        res_paths: list[Path] = list((self.data_dir / "res_map").iterdir())
        vol_paths: list[Path] = list((self.data_dir / "vol_map").iterdir())
        self.data_paths: list[tuple[Path, Path]] = list(zip(res_paths, vol_paths))
        self.len: int = len(self.data_paths)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x_path: Path
        y_path: Path
        x_path, y_path = self.data_paths[idx]
        x: np.ndarray[float] = np.loadtxt(x_path, delimiter=",").astype(np.float32)
        y: np.ndarray[float] = np.loadtxt(y_path, delimiter=",").astype(np.float32)
        x_torch: torch.Tensor = torch.tensor(x).unsqueeze(0) / 100
        y_torch: torch.Tensor = torch.tensor(y).unsqueeze(0) / 100
        return x_torch, y_torch


# %% randomimagedataset
class RandomDataset(Dataset):
    """
    given data_dir, which as res_map and vol_map subdirs, creates a dataset
    """

    def __init__(
        self,
        data_dir: Path,
        length,
        shape,
        process_n=10,
        R_top=3.0,  # 0.02 - 5
        R_bottom=0.02,  # 0.02 - 5
        x_path: Path | None = None,
        generate_x=True,
    ) -> None:
        self.process_n = process_n
        self.data_dir: Path = data_dir
        self.len: int = length
        self.shape: tuple[int, int] = shape
        self.res_path: Path = Path(self.data_dir) / str(self.shape)
        self.generate_x = generate_x
        self.og_x_path = x_path

        if not self.res_path.exists():
            self.res_path.mkdir(parents=True)

        self.x_path: Path = self.res_path / "x.npy"
        self.y_path: Path = self.res_path / "y.npy"

        self.pool = ThreadPool(processes=process_n)
        self.truss_mi_output = lambda x: r_middle_to_dv(
            self.shape[1],
            self.shape[0],
            x,
            R_top=R_top,
            R_bottom=R_bottom,
            R_ref=51.0,
            R_mux_top=10,
            R_mux_bottom=10,
            R_nop=4e10,
        )
        # vol_path: Path = self.data_dir / "vol_map"

        # if not self.res_path.exists():
        # pass
        # else:
        # self.x = np.load(self.x_path)
        # self.y = np.load(self.y_path)
        # self.len = len(self.x)
        # return
        #
        self.x: np.ndarray
        self.y: np.ndarray
        self.x, self.y = self.create_n_data(self.len)

        np.save(self.x_path, self.x)
        np.save(self.y_path, self.y)

    def create_n_data(self, n: int):
        if self.generate_x:
            x = np.random.randint(1, 7, (n, 1, *self.shape)) * 4
        else:
            x = np.load(self.og_x_path)
        x = x.astype(np.float32)
        y_agg: list[np.ndarray] = list()
        if self.process_n > 1:
            for y_i in tqdm(self.pool.map(self.truss_mi_output, x[:, 0])):
                y_agg.append(y_i)
        elif self.process_n == 1:
            for y_i in tqdm(map(self.truss_mi_output, x[:, 0])):
                y_agg.append(y_i)
                np.save(self.y_path, y_agg)

        y: np.ndarray = np.stack(y_agg)
        y: np.ndarray = np.expand_dims(y, 1)

        # x -= np.min(x)
        # x /= np.max(x)
        #
        # y -= np.min(y)
        # y /= np.max(y)

        return x, y

    def append_data_to_dataset(self, x, y) -> None:
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self.len = len(self.x)

        np.save(self.x_path, self.x)
        np.save(self.y_path, self.y)

    def __len__(self) -> int:
        return self.len

    def create_more(self, n):
        x, y = self.create_n_data(n)
        self.append_data_to_dataset(x, y)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x_torch: torch.Tensor = torch.tensor(self.x[idx])
        y_torch: torch.Tensor = torch.tensor(self.y[idx])
        return x_torch, y_torch


class GenerativeModel(nn.Module):
    def __init__(self, shape: tuple[int, int], layers: int):
        super().__init__()
        self.layers: int = layers
        self.height: int
        self.width: int
        self.height, self.width = shape

        self.horizontal_cnn_list = nn.ModuleList(
            [nn.Conv2d(1, 100, (1, self.width)) for i in range(self.layers)]
        )
        self.vertical_cnn_list = nn.ModuleList(
            [nn.Conv2d(1, 100, (self.height, 1)) for i in range(self.layers)]
        )

        self.vertical_tcnn_list = nn.ModuleList(
            [nn.ConvTranspose2d(100, 1, (self.height, 1)) for i in range(self.layers)]
        )
        self.horizontal_tcnn_list = nn.ModuleList(
            [nn.ConvTranspose2d(100, 1, (1, self.width)) for i in range(self.layers)]
        )

        self.lr = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor):
        past_output = self.batch_norm(x)
        for i in range(self.layers):
            # e.g. hv is horizontal then vertical
            hv = self.horizontal_tcnn_list[i](self.horizontal_cnn_list[i](past_output))
            vh = self.vertical_tcnn_list[i](self.vertical_cnn_list[i](past_output))
            layer_output = hv + vh
            past_output = layer_output + past_output
            past_output = self.lr(past_output)

        return past_output



class GenerativeLightning(L.LightningModule):
    def __init__(self, module, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.module = module
        self.lr = lr

    def training_step(self, batch, idx):
        x, y = batch
        y_pred = self.module(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log("loss", loss.item(), prog_bar=True)
        return loss

    def predict_step(self, batch, idx):
        x, y = batch
        return self.module(x), y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10_000)
        return [optimizer], [scheduler]



