import sys
import nest
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import lib.connectivity_landscape as cl
import lib.lcrn_network as lcrn
import glob
import new_sim_scripts.simulation_functions as sf 
import new_sim_scripts.test_sector_points_cartesian as dirconn


def connect_network_directions(neuron_parameters, synapse_parameters, AM, pop_all):
    [pop, popE, popI] = pop_all
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [iJ, dJ, g, x] = synapse_parameters
    [iso_AM, dir_AM] = AM 
	
    offsetE = popE[0]
    offsetI = popI[0]

    for idx in range(nE):
        iso_targets = np.where(iso_AM[idx,:] == 1.)[0] + offsetE
        dir_targets = np.where(dir_AM[idx,:] == 1.)[0] + offsetE
        nest.Connect([popE[idx]], iso_targets.tolist(), syn_spec={'weight': iJ})
        nest.Connect([popE[idx]], dir_targets.tolist(), syn_spec={'weight': dJ})

    for idx in range(nI):
        iso_targets = np.where(iso_AM[idx + nE,:] == 1.)[0] + offsetE
        dir_targets = np.where(dir_AM[idx + nE,:] == 1.)[0] + offsetE
        nest.Connect([popI[idx]], iso_targets.tolist(), syn_spec={'weight': -g * iJ})
        nest.Connect([popI[idx]], dir_targets.tolist(), syn_spec={'weight': -g * dJ})

    return

def create_network_directions(neuron_parameters, synapse_parameters, AM):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    #[Je, g, x] = synapse_parameters 
        
    popE = nest.Create(neuron_type, nE, params=neuron_paramsE)
    popI = nest.Create(neuron_type, nI, params=neuron_paramsI)

    pop = popE + popI
    pop_all = [pop, popE, popI]

    connect_network_directions(neuron_parameters, synapse_parameters, AM, pop_all)
    
    return pop_all 


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

    J = float(sys.argv[4])
    g = float(sys.argv[5])
    synapse_parameters = [J, g]

    # connectivity parameters

    p = [pEE, pEI, pIE, pII] = [0.05, 0.05, 0.05, 0.05]
    shift = 1.
    seed = int(sys.argv[8])
    landscape_type = sys.argv[7]
    landscape_size = int(sys.argv[2])
    asymmetry = sys.argv[1]
    std = [stdEE, stdEI, stdIE, stdII] = [float(x) for x in sys.argv[3].split(',')]
    connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, std, shift, seed]
    AM_address = '/home/bhalla/shreyal/scripts/AM_all/'
    fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_shift{}_base{}'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, shift, seed)
    save_AM_parameters = [AM_address, fname]

    # simulation parameters

    t_sim = 2000.
    t_warmup = 100.
    t_step = 0.1
    nfiles = 4 
    output_file = 'EI_spiketimes_' + asymmetry
    noise = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = [float(x) for x in sys.argv[6].split(',')]
    simulation_parmeters = [t_warmup, t_sim, noise, output_file]

    # running simulation

    #np.random.seed(0)
    nest.ResetKernel()
    rng_seeds = np.random.choice(np.arange(10000), int(nfiles)).tolist()
    nest.SetKernelStatus({
        'rng_seeds': rng_seeds,
        'local_num_threads': int(nfiles),
        'resolution': 0.1,
        'data_path': 'new_sim_scripts/EI_{}/Data'.format(asymmetry),
        'overwrite_files': True,
        })
    
    files = glob.glob(AM_address + fname + '*.npz')

    if len(files) == 0:
        [AM, landscape, nrowL] = sf.create_connectivity_EI(neuron_parameters, connectivity_parameters, save_AM_parameters)
    else:
        [AM, landscape, nrowL] = sf.read_AM_normal_sparse(save_AM_parameters)
    
    pop_network = sf.create_network(neuron_parameters, synapse_parameters, AM)
    del([AM])
    ids, ts = sf.run_simulation_EI(pop_network, simulation_parmeters)    
   
    # save the landscape   
    plt.imshow(landscape.reshape(nrowL,nrowL), origin='lower', cmap='twilight', vmin=0, vmax=8)
    plt.xlabel('Neuron grid x-axis', fontsize='x-large')
    plt.ylabel('Neuron grid y-axis', fontsize='x-large')
    cbar = plt.colorbar()
    cbar.set_label('Asymmetry directions', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('new_sim_scripts/EI_dir_{}/landscape'.format(asymmetry))
    #plt.show()

    del([landscape, nrowL, ids, ts, cbar, pop_network])
