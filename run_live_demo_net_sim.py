from multiprocessing.connection import Listener

from src.looming_sim.emd.simulator_EMD import EMD_model# as Simulator
from src.looming_sim.lgmd.simulator_LGMD import LGMD_model# as Simulator
from src.looming_sim.format_spike_data import tiled_events

from src.looming_sim.emd.network_settings import params as params_emd
from src.looming_sim.lgmd.network_settings import params as params_lgmd

import numpy as np

from mpl_multiproc import NBPlot

#Simulator = LGMD_model
#params = params_lgmd

Simulator = EMD_model
params = params_emd

DT_GENN_MS = 1.0
DT_CARLA = 0.01

N_SUBDIV = 2
HALF_STRIDE = True

WIDTH, HEIGHT = 304, 240

p = params.copy()
N_SIM_STEPS = int(DT_CARLA * 1000 / DT_GENN_MS)
p["NT_MAX"] = N_SIM_STEPS * 2
p["HALF_STEP_TILES"] = HALF_STRIDE
p["N_SUBDIV_X"] = N_SUBDIV
p["N_SUBDIV_Y"] = N_SUBDIV
p["DT_MS"] = DT_GENN_MS
N_TILES = (N_SUBDIV * 2 - 1) if HALF_STRIDE else N_SUBDIV

multiplot = NBPlot(mode="line", n_lines=3)


simulator = Simulator(p)

plot_img = NBPlot(mode="image", shape=(simulator.S_height, simulator.S_width), vmin=-1e-5, vmax=1e-5)

voltages = np.zeros((N_TILES, N_TILES))

address = ('localhost', 6000)
listener = Listener(address)

conn = listener.accept()

print('connection accepted from', listener.last_accepted)

while True:


    #evts = conn.recv()
    '''
    if isinstance(evts, str) and evts == "close":
        break
    # set times to start at 0
    evts["t"] -= evts["t"].min()
    # NANOSECONDS to milliseconds
    evts["t"] = evts["t"] // 1000000

    evts = evts[evts["t"] < p["NT_MAX"] * DT_GENN_MS]

    x = evts["x"].astype("<u2")
    y = evts["y"].astype("<u2")
    t = evts["t"].astype("<u4")
    pol = evts["p"].astype("<u2")
    pol[:] = 1

    evts = np.array(list(zip(t, x, y, pol)), dtype=[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")])
    
    tld_evts = tiled_events(evts, WIDTH, HEIGHT, N_SUBDIV, N_SUBDIV, HALF_STRIDE)
    x = [tl_data[-1]["x"] for tl_data in tld_evts]
    y = [tl_data[-1]["y"] for tl_data in tld_evts]
    st = [tl_data[-1]["t"] for tl_data in tld_evts]
    pol = [tl_data[-1]["p"] for tl_data in tld_evts]

    spike_bitmasks, pol_bitmasks = simulator.generate_bitmasks(x, y, st, pol)

    # set the simulator's timestep to 0 so that it matches the incoming data
    simulator.model.timestep = 0
    simulator.spk_bm = spike_bitmasks
    simulator.pol_bm = pol_bitmasks

    # push the data to the device
    simulator.push_input_data_to_device()
    '''
    #for i in range(N_SIM_STEPS):
    #    simulator.model.step_time()

    for i in range(N_TILES):
        for j in range(N_TILES):
            '''
            #simulator.LGMD[i * N_TILES + j].pull_var_from_device("V")
            simulator.OUT[i * N_TILES + j].pull_var_from_device("V")
            #voltages[i, j] = simulator.LGMD[i * N_TILES + j].vars["V"].view[0]
            voltages[i, j] = simulator.OUT[i * N_TILES + j].vars["V"].view[0]

            if i==1 and j==1:
                
                simulator.OUT[i * N_TILES + j].pull_var_from_device("r_left")
                r_left = simulator.OUT[i * N_TILES + j].vars["r_left"].view[0]
                simulator.OUT[i * N_TILES + j].pull_var_from_device("r_right")
                r_right = simulator.OUT[i * N_TILES + j].vars["r_right"].view[0]

                simulator.S[i * N_TILES + j].pull_var_from_device("vx")
                vx = np.reshape(simulator.S[i * N_TILES + j].vars["vx"].view[:], (simulator.S_height, simulator.S_width))

                multiplot.plot(np.array([voltages[i, j], r_left, r_right]))
                plot_img.plot(vx)
            '''
            pass            

    #print(voltages)

    #conn.send(voltages)

listener.close()