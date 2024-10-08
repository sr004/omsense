from lcapy import *
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy import lambdify

from lcapy import *
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy import lambdify
from scipy.ndimage import convolve

def gaussian_kernel(size, sigma=1.0, emphasis_factor=1.0):
    kernel_range = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize the kernel
    # Apply emphasis on higher values
    kernel = kernel ** (1 / emphasis_factor)
    return kernel / kernel.sum()

def gaussian_blur(image, kernel_size=1, sigma=1.0, emphasis_factor=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = convolve(image, kernel, mode='reflect')
    return blurred_image

def make_array(
    w: int,
    h: int,
    r1: int,r2,
    rmid:int, # r_mid
    rmux1:int,
    rmux2:int,
    active,
) -> Circuit:
    output_circuit = Circuit()
    # created top layer
    for y in range(h):
        output_circuit.add(f"R_top{y} t{y} t{y}_{0} {rmux1};right")
        for x in range(1, w):
            output_circuit.add(f"R_top{y}_{x} t{y}_{x-1} t{y}_{x} {r1};right")
    # create bottom layer
    for x in range(w):
        for y in range(h):
            # output_circuit.add(f"R_middle{y}{x} t{y}{x} b{y}{x} {rmid};rotate=-45")
            if y < h - 1:
                output_circuit.add(f"R_bottom{y}_{x} b{y}_{x} b{y + 1}_{x},{r2};down")

        output_circuit.add(f"R_bottom{x} b{h - 1}_{x} b{x} {rmux2};down")

    # find activated
    active_indices = np.transpose(np.indices(active.shape), (1, 2, 0))
    active_mask = active > 0
    active_values = active[active_mask]
    active_locations = active_indices[active_mask]

    for loc, val in zip(active_locations, active_values):
        output_circuit.add(f"R_middle{loc[0]}_{loc[1]} t{loc[0]}_{loc[1]} b{loc[0]}_{loc[1]} {val};rotate=-45")

    # syntax sugar
    output_circuit.add(";node_spacing=4, scale=0.5")
    return output_circuit


def r_middle_to_dv(
    height: int,
    width: int,
    x: np.ndarray[np.ndarray[float]],
    R_top: float,
    R_bottom: float,
    R_ref: float,
    R_mux_top: float,
    R_mux_bottom: float,
    R_nop: float,
):
    tqtq = tqdm(range(height * width))
    out=np.zeros((height,width))
    for i in range(height):
        tqtq.refresh()
        for j in range(width):
            circuit = make_array(
                height,
                width,
                R_top,
                R_bottom,
                R_nop,
                R_mux_top,
                R_mux_bottom,
                x
            )
            circuit.add(f'W? t{i} t1_1;left')
            circuit.add(f'V t1_1 b0_1 5;down=4')
            circuit.add(f'R_ref b{j} b0_1 {R_ref};left=3.7')
            # circuit.solve_method = 'GE'
            # print(circuit)
            for unconnected_node in circuit.unconnected_nodes():
                if "_" not in unconnected_node:
                    if unconnected_node[0] == "t":
                        circuit.remove("R_top" + unconnected_node[1:])
                    elif unconnected_node[0] == "b":
                        circuit.remove("R_bottom" + unconnected_node[1:])
                else:
                    print(unconnected_node)
            # circuit.draw("temp.png")
            # av=circuit['R_ref'].v.subs({'R_ref': R_ref}).evaluate()
            try:
                f = lambdify((), circuit['R_ref'].v.subs({'R_ref': R_ref}).sympy, "jax")
                av = f()
                out[i,j]=av
            except:
                out[i,j]=0.0
            tqtq.update()
            # tqtq.refresh()
    return np.array(out)

# if __name__ == '__main__':
    # height,width=8,8
    # R1,R2=3,0.02
    # R_op,Rmux1,Rmux2=400000000000000,10,10
    # Rref=51
    # values={'Rref':51}
    # Ract=20
# 
    # for sample in range(554, 1000):
        # ex_input = np.random.rand(height,width) - 0.7
        # ex_input[:] = np.where(ex_input <= 0, 0, ex_input)
        # for mid_res in range(4,25,2):
            # ex_input[:] = np.where(ex_input > 0, mid_res, ex_input)
            # a=ex_input[:]
            # np.savetxt(f'/Users/shubhamrohal/artifact_origami/data/simulation/10x10/ours/res_map/res_{sample}_{mid_res}.csv',ex_input,delimiter=',')
            # ex_input=gaussian_blur(ex_input,emphasis_factor=0.5)
            # ex_output = r_middle_to_dv(
                # ex_input.shape[1],
                # ex_input.shape[0],
                # ex_input,
                # R_top=3.0,
                # R_bottom=0.02,
                # R_ref=51.0,
                # R_mux_top=10,
                # R_mux_bottom=10,
                # R_nop=100
            # )
# 
            # 
            # np.savetxt(f'/Users/shubhamrohal/artifact_origami/data/simulation//10x10/ours/vol_map/vol_{sample}_{mid_res}.csv',ex_output,delimiter=',')
           # 
