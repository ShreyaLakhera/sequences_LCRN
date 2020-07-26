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
from subprocess import call
from sklearn.cluster import DBSCAN
import json

def run_simulation_EI_time(pop_all, simulation_parameters):
    
    [pop, popE, popI] = pop_all
    [t_warmup, t_burst, t_sim, noise, output_file] = simulation_parameters
    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = noise

    noise_params_E = {'mean': noise_mean_E, 'std': noise_std_E}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}
    reset_params = {'mean': 0., 'std': 0.}

    noiseE = nest.Create('noise_generator', params=noise_params_E)
    noiseI = nest.Create('noise_generator', params=noise_params_I)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)
    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    
    nest.SetStatus(noiseE, params=noise_params_E)
    nest.SetStatus(noiseI, params=noise_params_I)
    nest.Simulate(t_burst)

    nest.SetStatus(noiseE, params=reset_params)
    nest.SetStatus(noiseI, params=reset_params)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def run_simulation_EI_serial(pop_all, simulation_parameters, AB_all):
    
    [pop, popE, popI] = pop_all
    [t_warmup, del_T, noise, output_file] = simulation_parameters
    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = noise
    [idA, listA, idB, extra_params, idB_params] = AB_all

    noise_params_E = {'mean': noise_mean_E, 'std': noise_std_E}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}
    reset_params = {'mean': 0., 'std': 0.}

    noiseE = nest.Create('noise_generator', params=noise_params_E)
    noiseI = nest.Create('noise_generator', params=noise_params_I)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    listB = []
    timeAB = []
    
    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)
    nest.Connect(pop, sd)

    if idA in ['g', 'J', 'g2', 'pEE', 'pEI', 'pIE', 'pII', 'pEE1', 'pEI1', 'pIE1', 'pII1']:
        print('Reading connections...')

    if idA[:3] == 'pEE':
        conn = nest.GetConnections(popE, popE)
    elif idA[:3] == 'pEI':
        conn = nest.GetConnections(popE, popI)
    elif idA[:3] == 'pIE':
        conn = nest.GetConnections(popI, popE)
    elif idA[:3] == 'pII':
        conn = nest.GetConnections(popI, popI)
    elif idA == 'g':
        conn = nest.GetConnections(popI)
    elif idA == 'J':
        conn = nest.GetConnections(pop, pop)
    elif idA == 'g2':
        G1, G2 = np.array(popE), np.array(popE)
        N1, N2 = extra_params[1][:50], extra_params[2][:50]
        if asymmetry[0] == 'I':
            G1 = np.array(popI)
        if asymmetry[1] == 'I':
            G2 = np.array(popI)
        conn = nest.GetConnections(tuple(G1[N1]), tuple(G2[N2]))
        print(type(conn), len(conn))
        del(N1,N2,G1,G2)
        
    if idA in ['g', 'J', 'g2', 'pEE', 'pEI', 'pIE', 'pII', 'pEE1', 'pEI1', 'pIE1', 'pII1']:
        print("Reading connections done.")
        init_wval = nest.GetStatus(conn, 'weight')
        print("Reading initial weights done.")
        extra_params = extra_params + [conn, init_wval]

    nest.Simulate(t_warmup)
        
    for iT,valA in enumerate(listA):
        
        if idA == 'input':
            paramA = [valA, extra_params[0], noiseE]
        else:
            paramA = [valA] + extra_params + [pop_all]
        
        updateA(idA, paramA)
        nest.Simulate(del_T)
        
        sd_stat = nest.GetStatus(sd, 'events')[0]
        t_now = t_warmup + (iT)*del_T
        ts_ix = np.where(sd_stat['times'] > t_now)
        ids = sd_stat['senders'][ts_ix]
        ts = sd_stat['times'][ts_ix]

        if len(ts) > 0:
            valB = recordB(idB, ids, ts, idB_params)
        else:
            valB = -1        # 0 or -1, some distinguishable value
        listB.append(valB)
        t_now = t_warmup + iT*del_T
        timeAB.append(t_now)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts, listB, timeAB

def updateA(idA, paramA):

    if idA == 'input':
        [nm, ns, noise_device] = paramA
        params = {'mean': float(nm), 'std': float(ns)}
        nest.SetStatus(noise_device, params=params)
        
    elif idA[:1] == 'p':
        [p, initial_p, all_conn, init_wval, pop_all] = paramA
        [pop, popE, popI] = pop_all
        print([type(init_wval), np.shape(init_wval)])
        print("setting status of {} synapses".format(len(all_conn)))
        nest.SetStatus(all_conn, 'weight', list(init_wval))
        print("Done \n")
        conn_which = np.random.choice(np.arange(len(all_conn)), int(len(all_conn)*(initial_p-p))).astype(int)
        print(['Number of connections', type(conn_which), np.shape(conn_which)])
        new_conn = tuple(np.array(all_conn)[conn_which])
        if len(new_conn) > 0:
            print("Set some to zero")
            nest.SetStatus(new_conn, 'weight', 0.) # set zero
        
    elif idA == 'g':
        [g, init_g, connI, init_I_weights, pop_all] = paramA
        print("\ng = {}".format(g))
        new_I_weights = np.array(init_I_weights).astype(float)  * (g/init_g)
        print("Setting new I weights...")
        nest.SetStatus(connI, 'weight', new_I_weights)
        print("Done.")
        
    elif idA == 'J':
        [J, init_J, conn, init_weights, pop_all] = paramA
        print("\nJ = {}".format(J))
        new_weights = np.array(init_weights) * (J/init_J)
        print("Setting new weights...")
        nest.SetStatus(conn, 'weight', new_weights)
        print("Done.")
        
    elif idA == 'g2':
        [g2, init_g2, N1, N2, conn, init_weights, pop_all] = paramA
        print("\ng2 = {}".format(g2))
        new_weights = np.array(init_weights) * (g2/init_g2)
        print("Setting new weights...")
        nest.SetStatus(conn, 'weight', new_weights)
        print("Done.")

    elif idA == 'Vth':
        [vth, init_vth, pop_all] = paramA
        [pop, popE, popI] = pop_all
        print("\nV_th = {}".format(vth))
        nest.SetStatus(pop, 'V_th', vth)
    
    elif idA == 'VthE':
        [vth, init_vth, pop_all] = paramA
        [pop, popE, popI] = pop_all
        print("\nV_th = {}".format(vth))
        nest.SetStatus(popE, 'V_th', vth)
    
    elif idA == 'VthI':
        [vth, init_vth, pop_all] = paramA
        [pop, popE, popI] = pop_all
        print("\nV_th = {}".format(vth))
        nest.SetStatus(popI, 'V_th', vth)
    
    elif idA == 'taumall':
        [taum, init_taum, pop_all] = paramA
        print("\ntau_m = {}".format(taum))
        [pop, popE, popI] = pop_all
        nest.SetStatus(pop, 'tau_m', taum)
    
    elif idA == 'taumE':
        [taum, init_taum, pop_all] = paramA
        print("\ntaumE = {}".format(taum))
        [pop, popE, popI] = pop_all
        nest.SetStatus(popE, 'tau_m', taum)
    
    elif idA == 'taumI':
        [taum, init_taum, pop_all] = paramA
        print("\ntaumI = {}".format(taum))
        [pop, popE, popI] = pop_all
        nest.SetStatus(popI, 'tau_m', taum)

    return 

def recordB(idB, ids, ts, idB_params):
    
    if idB == 'avg_FR':
        [nE, nI, del_T] = idB_params
        t_start = np.min(ts)
        #firing_rate, e = np.histogram2d(ts, ids, bins=[np.arange(t_start + 450, t_start + 500., 10.), range(nE+nI+1)])[0]
        FR_time = pf.get_FR_time(ts, ids, nE + nI, 20., t_start, t_start + del_T)
        valB = np.mean(FR_time[-5:,:nE]) / 100.
    
    if idB == 'n_bumps':
        [nrowE, ncolE, eps, min_samples] = idB_params
        t_step = 20.
        t_start = np.min(ts)
        t_end = np.max(ts)
        idsE = ids[ids < nrowE*ncolE + 1]
        tsE = ts[ids < nrowE*ncolE + 1]
        FR_time = pf.get_FR_time(tsE, idsE, nrowE*ncolE, t_step, t_start, t_end)
        nclus_all = []
        #print(len(tsE))
        for i in range(np.shape(FR_time)[1]):
        #for i in range(5):
            fN = np.where(FR_time[:,i] > 0)[0]
            nclusters = 0
            if len(fN) > 10:
                xNs, yNs = fN // nrowE, fN % ncolE
                data = np.vstack((xNs, yNs)).T
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
                labels = db.labels_
                nclusters = np.max(labels) + 1
            #print(['Clusters', len(fN), nclusters])
            nclus_all.append(nclusters)    
        valB = np.mean(nclus_all[-5:])
        print("\nNumber of clusters = {}".format(valB))

    return valB

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
    alpha = 1. 
    landscape_type = 'perlinuniform' 
    landscape_size = 3
    asymmetry = 'EI'
    std = [stdEE, stdEI, stdIE, stdII] = [7, 9, 7, 9]
    connectivity_parameters = [landscape_type, landscape_size, asymmetry, p, shift, std, alpha, seed]
    AM_address = '/home/bhalla/shreyal/scripts/AM_all/'
    fname = 'size{}_{}_{}_std{}-{}-{}-{}_p{}-{}-{}-{}_shift{}_alpha{}_base{}_'.format(landscape_size, asymmetry, landscape_type, stdEE, stdEI, stdIE, stdII, pEE, pEI, pIE, pII, shift, alpha, seed)
    save_AM_parameters = [AM_address, fname]

    # simulation parameters

    del_T = 1000.
    t_warmup = 500.
    t_step = 0.1
    nfiles = 1 
    output_file = 'EI_spiketimes_' + asymmetry
    noise = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = [400., 100., 400., 100.]
    simulation_parmeters = [t_warmup, del_T, noise, output_file]


    # range of A and other parameters
    
    idA = 'Vth'
    idB = 'n_bumps'
    ntrials = 20
    
    listA_all = {
            "input" : [200.,250.,300.,350.,400.,450.,500.,550.,600.,650.,700.],
            "pEE" : [0.04, 0.042, 0.044,0.046,0.048,0.05],
            "pEI" : [0.04, 0.042, 0.044,0.046,0.048,0.05],
            "pIE" : [0.04, 0.042, 0.044,0.046,0.048,0.05],
            "pII" : [0.04, 0.042, 0.044,0.046,0.048,0.05],
            "pEE1" : [0, 0.01, 0.02,0.03,0.04,0.05],
            "pEI1" : [0, 0.01, 0.02,0.03,0.04,0.05],
            "pIE1" : [0, 0.01, 0.02,0.03,0.04,0.05],
            "pII1" : [0, 0.01, 0.02,0.03,0.04,0.05],
            "g" : [5,6,7,8,9,10,11,12,13,14,15],
            "J" : [5,7.5,10,12.5,15,17.5,20,22.5,25],
            "g2" : [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3],
            "alpha" : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
            "Vth": [-65., -60., -55., -50., -45., -40., -35., -30.],
            "VthE": [-65., -60., -55., -50., -45., -40., -35., -30.],
            "VthI": [-65., -60., -55., -50., -45., -40., -35., -30.],
            "taumall": [4.,6.,8.,10.,12.,14.,16.,18.,20.,22.,24.,26.,28.,30.],
            "taumE": [4.,6.,8.,10.,12.,14.,16.,18.,20.,22.,24.,26.,28.,30.],
            "taumI": [4.,6.,8.,10.,12.,14.,16.,18.,20.,22.,24.,26.,28.,30.]
            }

    listA = listA_all[idA]

    Aparams = {
            "input" : [noise_std_E],
            "pEE" : [pEE],
            "pEI" : [pEI],
            "pIE" : [pIE],
            "pII" : [pII],
            "pEE1" : [pEE],
            "pEI1" : [pEI],
            "pIE1" : [pIE],
            "pII1" : [pII],
            "g" : [g],
            "J" : [J],
            "g2" : [g2],
            "alpha" : [],
            "Vth" : [neuron_paramsE["V_th"]],
            "VthE" : [neuron_paramsE["V_th"]],
            "VthI" : [neuron_paramsE["V_th"]],
            "taumall" : [neuron_paramsE["tau_m"]],
            "taumE" : [neuron_paramsE["tau_m"]],
            "taumI" : [neuron_paramsE["tau_m"]],
            }
    

    eps = float(sys.argv[1])

    Bparams = {
            "avg_FR": [nE, nI, del_T], 
            "n_bumps": [nrowE, ncolE, eps, 10]        # [nrowE, ncolE, eps, min_samples]
            }

    idA_params = Aparams[idA]
    idB_params = Bparams[idB]

    # Address for saving data
    saving_address = "/home/bhalla/shreyal/data/EI_brunel_{}/EI_serial_{}_{}_size{}_std{}-{}-{}-{}_J{}_g{}g2{}_nmE{}I{}_alpha{}_base{}_taum{}_pEE{}_A{}_B{}_{}trial_{}delT_eps{}/".format(asymmetry, asymmetry, landscape_type, landscape_size, stdEE, stdEI, stdIE, stdII, int(J), int(g), int(g2), int(noise_mean_E), int(noise_mean_I), alpha, seed, int(neuron_paramsE["tau_m"]), pEE, idA, idB, ntrials, del_T, eps)
    call("mkdir {}".format(saving_address), shell=True)
    output_file1 = 'EI_spiketimes_{}'.format(asymmetry)
    
    # read the AM
    
    files = glob.glob(AM_address + fname + '*brunel_old_asym*.npz')

    if len(files) == 0:
        [ran_AM, dir_AM, landscape, nrowL] = sf.create_connectivity_EI_brunel_old_asymm(neuron_parameters, connectivity_parameters, save_AM_parameters)
    else:
        [ran_AM, dir_AM, landscape, nrowL] = sf.read_AM_brunel_old_asymm_sparse(save_AM_parameters)

    if idA == 'g2':
        if asymmetry == 'EE':
            (N1, N2) = np.where(dir_AM[:nE, :nE] > 0)
        elif asymmetry == 'EI':
            (N1, N2) = np.where(dir_AM[:nE, nE:] > 0)
        elif asymmetry == 'IE':
            (N1, N2) = np.where(dir_AM[nE:, :nE] > 0)
        elif asymmetry == 'II':
            (N1, N2) = np.where(dir_AM[nE:, nE:] > 0)
        print(type(N1), len(N1))
        idA_params += [N1, N2]

    # running simulation - increasing 

    listB0_all = []
    
    if idA == 'input':
        noise_mean_E = listA[0]
        noise_mean_I = noise_mean_E
        noise = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] 
        simulation_parmeters = [t_warmup, del_T, noise, output_file]

    for nt in range(ntrials):
        output_file = 'EI_spiketimes_{}_{}_{}'.format(asymmetry, 'increasing', nt)
        simulation_parmeters = [t_warmup, del_T, noise, output_file]
        nest.ResetKernel()
        rng_seeds = np.random.choice(np.arange(10000), int(nfiles)).tolist()
        nest.SetKernelStatus({
            'rng_seeds': rng_seeds,
            'local_num_threads': int(nfiles),
            'resolution': 0.1,
            'data_path': saving_address,
            'overwrite_files': True,
            })
        pop_network = sf.create_network_directions(neuron_parameters, synapse_parameters, [ran_AM, dir_AM])
        ids0, ts0, listB0, timeAB0 = run_simulation_EI_serial(pop_network, simulation_parmeters, [idA, listA, idB, idA_params, idB_params])
        listB0_all.append(listB0)
        
        output_file2 = '{}{}_increasing_{}_listB0_all.json'.format(saving_address, output_file1, nt)
        wfile = open(output_file2, 'w')
        json.dump(listB0, wfile)
        wfile.close()
    
    listA0 = listA.copy()

    # running simulation - decreasing 
    listA.reverse()
    listB1_all = []

    if idA == 'input':
        noise_mean_E = listA[0]
        noise_mean_I = noise_mean_E
        noise = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] 
        simulation_parmeters = [t_warmup, del_T, noise, output_file]

    for nt in range(ntrials):
        output_file = 'EI_spiketimes_{}_{}_{}'.format(asymmetry, 'decreasing', nt)
        simulation_parmeters = [t_warmup, del_T, noise, output_file]
        nest.ResetKernel()
        rng_seeds = np.random.choice(np.arange(10000), int(nfiles)).tolist()
        nest.SetKernelStatus({
            'rng_seeds': rng_seeds,
            'local_num_threads': int(nfiles),
            'resolution': 0.1,
            'data_path': saving_address,
            'overwrite_files': True,
            })

        pop_network = sf.create_network_directions(neuron_parameters, synapse_parameters, [ran_AM, dir_AM])
        ids1, ts1, listB1, timeAB1 = run_simulation_EI_serial(pop_network, simulation_parmeters, [idA, listA, idB, idA_params, idB_params])    
        listB1_all.append(listB1)
        
        output_file2 = '{}{}_decreasing_{}_listB1_all.json'.format(saving_address, output_file1, nt)
        wfile = open(output_file2, 'w')
        json.dump(listB1, wfile)
        wfile.close()

    listA1 = listA.copy()
    listB0_mean = np.mean(np.array(listB0_all), axis=0)
    listB1_mean = np.mean(np.array(listB1_all), axis=0)
    listB0_std = np.std(np.array(listB0_all), axis=0)
    listB1_std = np.std(np.array(listB1_all), axis=0)
    print([np.shape(listA1), np.shape(listB1_mean), np.shape(listB1_std)])

    # Plots

    # Save the landscape   
    plt.imshow(landscape.reshape(nrowL,nrowL), origin='lower', cmap='twilight', vmin=0, vmax=8)
    plt.xlabel('Neuron grid x-axis', fontsize='x-large')
    plt.ylabel('Neuron grid y-axis', fontsize='x-large')
    cbar = plt.colorbar()
    cbar.set_label('Asymmetry directions', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('{}landscape'.format(saving_address))
    #plt.show()
   
    """
    # Plot A and B
    fig, ax = plt.subplots()
    ax.plot(timeAB0, listA0, 'b-', label=idA)
    ax.plot(timeAB1, listA1, 'b--', label=idA)
    ax.set_ylabel(idA)
    ax.set_xlabel('Time')
    ax1 = ax.twinx()
    for i in range(ntrials):
        ax1.plot(timeAB0, listB0_all[i], 'r-', label=idB)
        ax1.plot(timeAB1, listB1_all[i], 'r--', label=idB)
    ax1.set_ylabel(idB)
    plt.tight_layout()
    plt.savefig('{}brunel_serial_AB_time_multiple'.format(saving_address))
    """

    # Plot A vs B
    fig, ax = plt.subplots()
    for i in range(ntrials):
        ax.plot(listA0, listB0_all[i], '-', color='#998ec3', alpha=0.2)
        ax.plot(listA1, listB1_all[i], '-', color='#f1a340', alpha=0.2)
    ax.errorbar(listA0, listB0_mean, yerr=listB0_std, fmt='-o', color='#998ec3', label='Increasing')
    ax.errorbar(listA1, listB1_mean, yerr=listB1_std, fmt='-o', color='#f1a340', label='Decreasing')
    ax.set_xlabel(idA)
    ax.set_ylabel(idB)
    ax.legend()
    plt.tight_layout()
    plt.savefig('{}brunel_serial_AB_multiple'.format(saving_address))

    plt.close('all')

    # Raster

    # Average firing rate
