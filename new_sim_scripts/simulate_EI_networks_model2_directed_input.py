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

    [J] = [float(x) for x in sys.argv[4].split(',')]
    [g, g2] = [float(x) for x in sys.argv[5].split(',')]
    synapse_parameters = [J, g, g2]

    # connectivity parameters

    p = [pEE, pEI, pIE, pII] = [0.05, 0.05, 0.05, 0.05]
    seed = int(sys.argv[10])
    separation = float(sys.argv[9]) 
    width = float(sys.argv[7])
    alpha = float(sys.argv[8]) #0.2
    landscape_type = 'perlinuniform'
    landscape_size = int(sys.argv[2])
    asymmetry = sys.argv[1]
    std = [stdEE, stdEI, stdIE, stdII] = [float(x) for x in sys.argv[3].split(',')]
    connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, std, separation, width, alpha, seed]
    AM_address = '/home/bhalla/shreyal/scripts/AM_all/'
    fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_alpha{}_sep{}_width{}_base{}_'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, alpha, separation, width, seed)
    save_AM_parameters = [AM_address, fname]

    # simulation parameters

    t_sim = 2000.
    t_burst = 2000.
    t_warmup = 100.
    t_step = 0.1
    nfiles = 4 
    output_file = 'EI_spiketimes_' + asymmetry
    inp_center_N = [cNx, cNy] = [int(x) for x in sys.argv[11].split(',')]
    width = int(sys.argv[12])
    noise = [nmE, nsE, nmI, nsI, nmP, nsP] = [float(x) for x in sys.argv[6].split(',')]
    simulation_parmeters = [t_warmup, t_burst, t_sim, noise, output_file, inp_center_N, width]

    # running simulation
    
    #np.random.seed(0)
    nest.ResetKernel()
    nest.SetKernelStatus({
        'local_num_threads': int(nfiles),
        'resolution': 0.1,
        'data_path': 'new_sim_scripts/EI_dir_{}/Data'.format(asymmetry),
        'overwrite_files': True,
        })
    
    files = glob.glob(AM_address + fname + '*isoAM.npz')

    if len(files) == 0:
        AM = [isoAM, dirAM, landscape, nrowL] = sf.create_connectivity_EI_dir(neuron_parameters, connectivity_parameters, save_AM_parameters)
    else:
        AM = [isoAM, dirAM, landscape, nrowL] = sf.read_AM_dir_sparse(save_AM_parameters)
        print('Multiple connections')
        print(np.shape(np.where(isoAM+dirAM > 1.)))
        print(np.where((isoAM + dirAM) > 1.))

    pop_network = sf.create_network_directions(neuron_parameters, synapse_parameters, [isoAM, dirAM])
    ids, ts = sf.run_simulation_EI_directed_input(pop_network, simulation_parmeters, neuron_parameters)    
   
    plt.imshow(landscape.reshape(nrowL,nrowL), origin='lower', cmap='twilight', vmin=0, vmax=8)
    plt.xlabel('Neuron grid x-axis', fontsize='x-large')
    plt.ylabel('Neuron grid y-axis', fontsize='x-large')
    cbar = plt.colorbar()
    cbar.set_label('Asymmetry directions', fontsize='x-large')
    plt.savefig('new_sim_scripts/EI_dir_{}/landscape'.format(asymmetry))
    #plt.show()
