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
import new_plot_scripts.plotting_functions as pf 
import new_sim_scripts.test_sector_points_cartesian as dirconn
from sklearn.cluster import DBSCAN
from subprocess import call
import json

def connect_network(neuron_parameters, synapse_parameters, AM, pop_all):
    [pop, popE, popI] = pop_all
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [J, g] = synapse_parameters
	
    offsetE = popE[0]
    offsetI = popI[0]
    
    for idx in range(nE):
        targets1 = np.where(AM[idx,:] == 1.)[0] + offsetE
        nest.Connect([popE[idx]], targets1.tolist(), syn_spec={'weight': J})

    for idx in range(nI):
        targets1 = np.where(AM[idx + nE,:] == 1.)[0] + offsetE
        nest.Connect([popI[idx]], targets1.tolist(), syn_spec={'weight': -g * J})

    return


def create_network(neuron_parameters, synapse_parameters, AM):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    #[Je, g, x] = synapse_parameters 
        
    popE = nest.Create(neuron_type, nE, params=neuron_paramsE)
    popI = nest.Create(neuron_type, nI, params=neuron_paramsI)

    pop = popE + popI
    pop_all = [pop, popE, popI]

    connect_network(neuron_parameters, synapse_parameters, AM, pop_all)
    
    return pop_all
    
def run_simulation(pop_all, simulation_parameters, neuron_parameters):
    
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, t_reset, noise_input_params, output_file, inp_pos, inp_width] = simulation_parameters
    [nmE, nmbE, nmpE, nmI, nmbI, nsE, nsI, nsp] = noise_input_params

    noise_params_E = {'mean': nmE, 'std': nsE}
    noise_params_I = {'mean': nmI, 'std': nsI}
    noise_params_bE = {'mean': nmbE, 'std': nsE}
    noise_params_bI = {'mean': nmbI, 'std': nsI}
    noise_params_pulse = {'mean': nmpE, 'std': nsp}
    reset_params = {'mean': 0., 'std': 0.}
    reset_params2 = {'mean': -1 * nmbE, 'std': 0.}

    noiseE = nest.Create('noise_generator')
    noiseI = nest.Create('noise_generator')
    #noisebE = nest.Create('noise_generator', params=reset_params)
    #noisebI = nest.Create('noise_generator', params=reset_params)
    noisep = nest.Create('noise_generator', params=reset_params)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})
    ncA = sf.get_grid_around(inp_pos, inp_width, nE, popE)

    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)
    nest.Connect(noisep, ncA)

    nest.Connect(pop, sd)

    # Simulate for warmup
    nest.Simulate(t_warmup)

    # Simulate with uniform input
    nest.SetStatus(noiseE, noise_params_E)
    nest.SetStatus(noiseI, noise_params_I)
    nest.Simulate(t_sim)

    """
    # Reset state
    nest.SetStatus(noiseE, reset_params)
    nest.SetStatus(noiseI, reset_params)
    nest.Simulate(t_reset)
    
    # Simulate with directed input
    nest.SetStatus(noiseE, noise_params_bE)
    nest.SetStatus(noiseI, noise_params_bI)
    nest.SetStatus(noisep, noise_params_pulse)
    nest.Simulate(t_sim)
    nest.SetStatus(noisep, reset_params)

    # Reset state
    nest.SetStatus(noiseE, reset_params2)
    nest.SetStatus(noiseI, reset_params2)
    nest.Simulate(t_reset)
    
    # Simulate with high uniform input
    nest.SetStatus(noiseE, noise_params_E)
    nest.SetStatus(noiseI, noise_params_I)
    nest.Simulate(t_sim)
    
    # Reset state
    nest.SetStatus(noiseE, reset_params2)
    nest.SetStatus(noiseI, reset_params2)
    nest.Simulate(t_reset)

    # Simulate with sparse uniform input
    r = 1.0
    #for _t in range(int(t_sim/100.)):
    for _t in range(int(t_sim/1000.)):
        #sim_neurons = tuple(np.array(popE)[np.random.choice(np.arange(0,len(popE), 1), int(len(popE)*r), replace=False).astype(int)])
        noisep2 = nest.Create("noise_generator", params=noise_params_bE)
        #nest.Connect(noisep2, sim_neurons)
        nest.Connect(noisep2, popE)
        #sim_neuronsI = tuple(np.array(popI)[np.random.choice(np.arange(0,len(popI), 1), int(len(popI)*r), replace=False).astype(int)])
        noisep2I = nest.Create("noise_generator", params=noise_params_bI)
        #nest.Connect(noisep2I, sim_neuronsI)
        nest.Connect(noisep2I, popI)
        nest.Simulate(1000.)
        nest.SetStatus(noisep2, reset_params)
        nest.SetStatus(noisep2I, reset_params)
        #del(noisep2, sim_neurons)#, noisep2I, sim_neuronsI)
        del(noisep2, noisep2I)
    
    # Reset state
    nest.SetStatus(noiseE, reset_params2)
    nest.SetStatus(noiseI, reset_params2)
    nest.Simulate(t_reset)

    # Simulate with tmeporally varying input - random points
    nest.SetStatus(noiseE, noise_params_bE)
    nest.SetStatus(noiseI, noise_params_bI)
    for _t in range(int(t_sim/100.)):
        inp_pos_t = [np.random.randint(nrowE), np.random.randint(ncolE)]
        ncAt = sf.get_grid_around(inp_pos_t, inp_width, nE, popE)
        noisep2 = nest.Create("noise_generator", params=noise_params_pulse)
        nest.Connect(noisep2, ncAt)
        nest.Simulate(100.)
        del(noisep2, ncAt, inp_pos_t)

    # Reset state
    nest.SetStatus(noiseE, reset_params2)
    nest.SetStatus(noiseI, reset_params2)
    nest.Simulate(t_reset)
    
    # Simulate with temporally varying input - interpolated curve
    nest.SetStatus(noiseE, noise_params_bE)
    nest.SetStatus(noiseI, noise_params_bI)
    xn1, yn1 = sf.get_interpolated_curve(nrowE, ncolE, 10)#int(t_sim/100.))
    for _t in range(int(t_sim/100.)):
        ncAt = sf.get_grid_around([xn1[_t], yn1[_t]], inp_width, nE, popE)
        noisep2 = nest.Create("noise_generator", params=noise_params_pulse)
        nest.Connect(noisep2, ncAt)
        nest.Simulate(100.)
        del(noisep2, ncAt)
    del(xn1,yn1)
    """    
    # Record spikes
    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def get_nbumps_ids(ts, ids, eps, min_samples, nrowE, ncolE, t_start, t_end, t_step):
    FR_time = pf.get_FR_time(ts, ids, nrowE*ncolE, t_step, t_start, t_end)
    nclus_all = np.zeros(np.shape(FR_time)[1])
    for i in range(np.shape(FR_time)[1]):
        fN = np.where(FR_time[:,i] > 0)[0]
        nclusters = 0
        if len(fN) > 10:
            xNs, yNs = fN // nrowE, fN % ncolE
            data = np.vstack((xNs, yNs)).T
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = db.labels_
            nclusters = np.max(labels) + 1
        nclus_all[i] = nclusters
    return nclus_all


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
    
    J = 10.
    g = 8.
    g2 = 1.
    synapse_parameters = [J, g, g2]

    # connectivity parameters

    p = [pEE, pEI, pIE, pII] = [0.05, 0.05, 0.05, 0.05]
    shift = 1.
    seed = 0 
    alpha = 1.0 
    landscape_type = 'perlinuniform' 
    landscape_size = 3
    asymmetry = 'EI'
    std = [stdEE, stdEI, stdIE, stdII] = [7, 9, 7, 9]
    connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, shift, std, alpha, seed]
    AM_address = '/home/bhalla/shreyal/scripts/AM_all/'
    fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_shift{}_alpha{}_base{}_'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, shift, alpha, seed)
    save_AM_parameters = [AM_address, fname]

    # simulation parameters

    t_warmup = 100.
    t_sim = 2000.
    t_reset = 500.
    t_step = 0.1
    nfiles = 1 
    output_file = 'EI_spiketimes_' + asymmetry
    noise = [nmE, nmbE, nmpE, nmI, nmbI, nsE, nsI, nsp] = [0., 0., 0., 0., 0., 100., 100., 0.] # [600., 290., 300., 600., 290., 100., 100., 0.]
    inp_width = 8
    inp_pos_all = {
            'EE': [45,45],
            'EI': [75,105],
            'IE': [45,75],
            'II': [75,45]
            }

    # running simulation

    for nmE in [700.]:#[500., 600., 700., 800., 900., 1000.]:
        for nmI in [850., 900., 950., 1000.]:#[0., 300., 400., 500., 600., 700., 800., 900., 1000., 1100., 1200.]:
            for trial in range(2):
                
                print(asymmetry, alpha, nmE, nmI)

                #nmbE = 300.
                #nmbI = 400.
                Vth = -50
                taumE = 10.
                taumI = 10.
                r = 1.0
                inp_pos = inp_pos_all[asymmetry]
                fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_shift{}_alpha{}_base{}_'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, shift, alpha, seed)
                data_path = '/home/bhalla/shreyal/data/EI_brunel_{}/EI_{}_size{}_std{}-{}-{}-{}_g{}g2{}_J{}_nmE{}I{}p{}bE{}bI{}_alpha{}_base{}_taumE{}I{}_Vth{}_ratio{}_100s_trial{}/'.format(asymmetry, asymmetry, landscape_size, stdEE, stdEI, stdIE, stdII,int(g), int(g2), int(J), int(nmE), int(nmI), int(nmpE), int(nmbE), int(nmbI), alpha, seed, int(taumE), int(taumI), int(Vth), r, trial)
                output_file = 'EI_spiketimes_' + asymmetry
                connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, shift, std, alpha, seed]
                save_AM_parameters = [AM_address, fname]
                noise = [nmE, nmbE, nmpE, nmI, nmbI, nsE, nsI, nsp]
                simulation_parameters = [t_warmup, t_sim, t_reset, noise, output_file, inp_pos, inp_width]

                call('mkdir {}'.format(data_path), shell=True)

                nest.ResetKernel()
                rng_seeds = np.random.choice(np.arange(10000), int(nfiles)).tolist()
                nest.SetKernelStatus({
                    'local_num_threads': int(nfiles),
                    'rng_seeds': rng_seeds,
                    'resolution': 0.1,
                    'data_path': data_path,
                    'overwrite_files': True,
                    })
                
                files = glob.glob(AM_address + fname + '*.npz')

                if len(files) == 0:
                    [ran_AM, dir_AM, landscape, nrowL] = sf.create_connectivity_EI_brunel_old_asymm(neuron_parameters, connectivity_parameters, save_AM_parameters)
                else:
                    [ran_AM, dir_AM, landscape, nrowL] = sf.read_AM_brunel_old_asymm_sparse(save_AM_parameters)
                
                # create network and simulate

                pop_network = sf.create_network_directions(neuron_parameters, synapse_parameters, [ran_AM, dir_AM])
                [pop, popE, popI] = pop_network
                nest.SetStatus(pop, 'V_th', float(Vth))
                ids, ts = run_simulation(pop_network, simulation_parameters, neuron_parameters)    
                print()
                print(len(ts), 'spikes found.\n')

                if len(ts) == 0:
                    continue

                # Plot landscape

                plt.imshow(landscape.reshape(nrowL,nrowL), origin='lower', cmap='twilight', vmin=0, vmax=8)
                plt.xlabel('Neuron grid x-axis', fontsize='x-large')
                plt.ylabel('Neuron grid y-axis', fontsize='x-large')
                cbar = plt.colorbar()
                cbar.set_label('Asymmetry directions', fontsize='x-large')
                plt.savefig('{}landscape'.format(data_path))
                #plt.show()
                plt.close()
                
                # Make raster plots and save
                offsetE = 1
                offsetI = nE + 1
                gidxE = ids - offsetE < nE
                tsE, gidsE = ts[gidxE], ids[gidxE]         # Excitatory population
                tsI, gidsI = ts[~gidxE], ids[~gidxE]       # Inhibitory population
                
                plt.figure(figsize=[7.5,5])
                plt.ylim([0,nN])
                plt.xlim([0.,np.max(ts)])
                plt.plot(tsE, gidsE, '|', markersize=3)
                plt.plot(tsI, gidsI, '|', markersize=3)
                plt.xlabel('Time (ms)', fontsize=15)
                plt.ylabel('Neuron ID', fontsize=15)
                plt.tight_layout()
                plt.savefig('{}raster_UI'.format(data_path))
                plt.close()

                # Calculate nbumps and save
                """
                eps = 3
                min_samples = 10

                idxE = ids < nE + 1
                tsE, idsE = ts[idxE], ids[idxE]

                t_step_nbumps = 10.

                nclus1 = get_nbumps_ids(tsE, idsE, eps, min_samples, nrowE, ncolE, t_warmup, t_warmup + t_sim, t_step_nbumps)
                nclus2 = get_nbumps_ids(tsE, idsE, eps, min_samples, nrowE, ncolE, t_warmup + t_sim, t_warmup + 2*t_sim, t_step_nbumps)
                nclus3 = get_nbumps_ids(tsE, idsE, eps, min_samples, nrowE, ncolE, t_warmup + 2*t_sim, t_warmup + 3*t_sim, t_step_nbumps)

                np.save("{}nclus1".format(data_path), nclus1)
                np.save("{}nclus2".format(data_path), nclus2)
                np.save("{}nclus3".format(data_path), nclus3)
                """

                # make animation
                tres = 10.
                tw, ts, tr = t_warmup, t_sim, t_reset
                text_all = [
                        ['Warming up', 0., tw],
                        ['Uniform input', tw, tw + ts]
                #        ['Reset', tw + ts, tw + ts + tr],
                #        ['Directed input', tw + ts + tr, tw + 2*ts + tr]
                #        ['Reset', tw + 2*ts + tr, tw + 2*ts + 2*tr],
                #        ['High uniform input', tw + 2*ts + 2*tr, tw + 3*ts + 2*tr]
                #        ['Reset', tw + 3*ts + 2*tr, tw + 3*ts + 3*tr],
                #        ['Sparse uniform input', tw + 3*ts + 3*tr, tw + 4*ts + 3*tr],
                #        ['Reset', tw + 4*ts + 3*tr, tw + 4*ts + 4*tr],
                #        ['Temporal - Random', tw + 4*ts + 4*tr, tw + 5*ts + 4*tr],
                #        ['Reset', tw + 5*ts + 4*tr, tw + 5*ts + 5*tr],
                #        ['Temporal - Interpolated', tw + 5*ts + 5*tr, tw + 6*ts + 5*tr]
                        ]

                if len(tsE) == 0:
                    print('no E spikes')
                    continue
                """
                a = pf.make_animation_old_label(tsE, gidsE - offsetE, nrowE, ncolE, nE, tres, text_all)
                print("Animation made.")
                print("Saving animation...")
                a.save(data_path + 'activity_anim_E_UI.mp4', fps=5, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
                print("Animation saved.")
                plt.close()

                if len(tsI) == 0:
                    print('no I spikes')
                    continue

                a = pf.make_animation_old_label(tsI, gidsI - offsetI, nrowI, ncolI, nI, tres, text_all)
                print("Animation made.")
                print("Saving animation...")
                a.save(data_path + 'activity_anim_I_UI.mp4', fps=5, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
                print("Animation saved.")
                plt.close()
                """
