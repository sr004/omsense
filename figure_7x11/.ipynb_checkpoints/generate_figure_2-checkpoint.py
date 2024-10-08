# %% prereqs
from lib import RandomDataset
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

# %% figure_2_setup

figure_2_p = Path(".")
# figure_2_p = Path("figure_2")
# if not figure_2_p.exists():
    # figure_2_p.mkdir(parents=True)

# r_top and r_bottom have the same range [0.02,5]
_resistances: np.ndarray = np.arange(0.0, 6.0)
_resistances[0] = 0.02
# display(_resistances)
_resistances = np.concatenate(np.transpose(np.stack(np.meshgrid(_resistances, _resistances)),(1, 2, 0)))
_x = Path("./x.npy")
_x_data = np.load(_x)
display(_resistances, _x_data.shape)
distances = np.array([x * len(_resistances) // 20 for x in range(20)])
distances = list(zip(distances, list(distances[1:]) + [len(_resistances)]))
display(distances)


def fig_2_generate(both):
    start, end = both
    for _r_top, _r_bottom in _resistances[start:end]:
        RandomDataset(
            figure_2_p / f"r_top={_r_top},r_bottom={_r_bottom}",
            0,
            (5, 5),
            generate_x=False,
            R_top=_r_top,
            R_bottom=_r_bottom,
            x_path=_x,
            process_n=1,
        )

# %% figure_2_run
with Pool(20) as p:
    p.map(fig_2_generate, distances)
