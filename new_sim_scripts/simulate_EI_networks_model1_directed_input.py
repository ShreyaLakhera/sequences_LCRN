import sys
import nest
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import lib.connectivity_landscape as cl
import lib.lcrn_network as lcrn

import new_sim_scripts.simulation_functions as sf
import new_plot_scripts.plotting_functions as pf

def run_simulation_dynamic_input(pop_all, simulation_parameters, neuron_parameters):	

    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, t_poke, noise_input_params, output_file, inp_center_N, w] = simulation_parameters
    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I, noise_mean_pulse, noise_std_pulse] = noise_input_params

    noise_params_E = {'mean': noise_mean_E, 'std': noise_std_E}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}
    noise_params_pulse = {'mean': noise_mean_pulse, 'std': noise_std_pulse}
    reset_params = {'mean': 0., 'std': 0.}

    noiseE = nest.Create('noise_generator', params=noise_params_E)
    noiseI = nest.Create('noise_generator', params=noise_params_I)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)

    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    #nest.SetStatus(noiseE, params = reset_params)
    #nest.SetStatus(noiseI, params = reset_params)

    ncA = get_grid_around(inp_center_N, w, nE, popE)
    sp_noise = nest.Create("noise_generator", params = noise_params_pulse)
    nest.Connect(sp_noise, ncA)
    nest.Simulate(t_poke)
    nest.SetStatus(sp_noise, params=reset_params)

    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def get_grid_around(inpN,w, nN, pop):
    [x0,y0] = inpN
    ncol = nrow = int(np.sqrt(nN))

    ncA = []
    ncN_x = int(x0)
    ncN_y = int(y0)
    r = int(w/2)
    for i in range(ncN_x - r,ncN_x + r):
      	for j in range(ncN_y - r,ncN_y + r):
            n = ((j * ncol) + i) % nN
            ncA.append(pop[n])
    
    return ncA



if __name__ == "__main__":
   
    # neuron parameters:
    
    nrowE = ncolE = 120
    nrowI = ncolI = 60
    nE = nrowE * ncolE
    nI = nrowI * ncolI
    nN = nE + nI
    neuron_type = "iaf_psc_alpha"
    neuron_paramsE = {
            "C_m": 250.0,
            "E_L": -70.0,
            "t_ref": 2.0,
            "tau_m": 10.0,
            "tau_minus": 20.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 5.0,
            "V_reset": -70.0,
            "V_th": -55.0,
            }
    neuron_paramsI = neuron_paramsE         # change later
    neuron_parameters = [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI]

    # synapse parameters

    Je = float(sys.argv[4])
    g = float(sys.argv[5])
    synapse_parameters = [Je, g]

    # connectivity parameters

    p = [pEE, pEI, pIE, pII] = [0.05, 0.05, 0.05, 0.05]
    shift = 1.0
    landscape_type = 'perlinuniform'
    landscape_size = int(sys.argv[2])
    asymmetry = sys.argv[1]
    seed = int(sys.argv[9])
    std = [stdEE, stdEI, stdIE, stdII] = [float(x) for x in sys.argv[3].split(',')]
    connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, std, shift, seed] 
    AM_address = '/home/bhalla/shreyal/scripts/AM_all/'
    fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_shift{}_base{}'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, shift, seed)
    save_AM_parameters = [AM_address, fname]

    # simulation parameters

    inp_center_N = [inp_x, inp_y] = sys.argv[7].split(',')
    w = int(sys.argv[8])
    t_sim = 2000.
    t_warmup = 200.
    t_step = 0.1
    t_poke = 100.
    nfiles = 4 
    output_file = 'EI_spiketimes_' + asymmetry
    noise_input_params = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I, noise_mean_pulse, noise_std_pulse] = [float(x) for x in sys.argv[6].split(',')]
    simulation_parmeters = [t_warmup, t_sim, t_poke, noise_input_params, output_file, inp_center_N, w]
    simulation_parmeters0 = [t_warmup, t_sim, noise_mean_E, noise_std_E, output_file]
    

    # running simulation

    nest.ResetKernel()
    rng_seeds = np.random.choice(np.arange(10000), int(nfiles)).tolist()
    nest.SetKernelStatus({
        'rng_seeds': rng_seeds,
        'local_num_threads': int(nfiles),
        'resolution': 0.1,
        'data_path': 'new_sim_scripts/EI_dyninp_{}/Data'.format(asymmetry),
        'overwrite_files': True,
        })
    
    files = glob.glob(AM_address + fname + '*.npz')

    if len(files) == 0:
        [AM, landscape, nrowL] = sf.create_connectivity_EI(neuron_parameters, connectivity_parameters, save_AM_parameters)
    else:
        [AM, landscape, nrowL] = sf.read_AM_normal_sparse(save_AM_parameters)
    
    pop_network = sf.create_network(neuron_parameters, synapse_parameters, AM)
    ids, ts = run_simulation_dynamic_input(pop_network, simulation_parmeters, neuron_parameters)    
    #ids, ts = sf.run_simulation(pop_network, simulation_parmeters0)    

    # Plot firing rate
    offsetE = 1
    gidxE = ids - offsetE < nE
    tsE, gidsE = ts[gidxE], ids[gidxE]         # Excitatory population
    tsI, gidsI = ts[~gidxE], ids[~gidxE]       # Inhibitory population
    tsim = t_warmup + t_poke + t_sim
    firing_rate = pf.calculate_firing_rate(tsE, gidsE, nE, tsim)

    address = 'new_sim_scripts/EI_dyninp_{}/Data/'.format(asymmetry)
    plt.imshow(np.reshape(firing_rate, (nrowE, ncolE)), origin='lower')
    plt.xlabel('Neuron grid x-axis', fontsize='x-large')
    plt.ylabel('Neuron grid y-axis', fontsize='x-large')
    cbar = plt.colorbar()
    cbar.set_label('Average Firing Rate (spikes/ms)', fontsize='x-large')
    #plt.title('Network 2')
    plt.tight_layout()
    plt.savefig(address+'avg_fr')
