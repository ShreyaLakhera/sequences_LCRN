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

def read_simulation_EI_serial(simulation_parameters, AB_all):
    
    [t_warmup, del_T, noise, output_file] = simulation_parameters
    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = noise
    [idA, listA, idB, extra_params, idB_params] = AB_all

    listB = []
    timeAB = []
    print(output_file)
    ts_all, ids_all = pf.get_spiketimes2(output_file, nfiles)
    print("Read spiketimes all.")
    
    for iT,valA in enumerate(listA):
                     
        t_now = t_warmup + (iT)*del_T
        t_start = t_now
        t_end = t_now + del_T
        ts, ids = pf.getSpikesTimeInterval(ts_all, ids_all, t_start, t_end)       

        if len(ts) > 0:
            valB = recordB(idB, ids, ts, idB_params)
        else:
            print("No spiking!")
            valB = 0        # 0 or -1, some distinguishable value
        listB.append(valB)
        t_now = t_warmup + iT*del_T
        timeAB.append(t_now)

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
        print("Yet to be written!")
        [g2, init_g2, pop_all] = paramA
    
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
        idsE = ids[ids < nrowE*ncolE]
        tsE = ts[ids < nrowE*ncolE]
        t_start = np.min(tsE)
        t_end = np.max(tsE)
        FR_time = pf.get_FR_time(tsE, idsE, nrowE*ncolE, t_step, t_start, t_end)
        FR_threshold = 0.4 * np.mean(FR_time)
        nclus_all = []
        #print(len(tsE))
        for i in range(np.shape(FR_time)[1]):
        #for i in range(5):
            fN = np.where(FR_time[:,i] > FR_threshold)[0]
            nclusters = 0
            if len(fN) > 10:
                xNs, yNs = fN // nrowE, fN % ncolE
                data = np.vstack((xNs, yNs)).T
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
                labels = db.labels_
                nclusters = np.max(labels) + 1
                """
                print(nclusters)
                if nclusters > 10:
                    fig, ax = plt.subplots()
                    noiseN = np.where(labels == -1)[0]
                    cxns, cyns = xNs[noiseN], yNs[noiseN]
                    ax.plot(cxns, cyns, 'k.', label='Noise')
                    for _ci in range(nclusters):
                        cN = np.where(labels == _ci)[0]
                        cxns, cyns = xNs[cN], yNs[cN]
                        ax.plot(cxns, cyns, '.', label='cluster {}'.format(_ci))
                    ax.legend()
                    ax.set_xlim([0,120])
                    ax.set_ylim([0,120])
                    plt.show()
                """
            #print(['Clusters', len(fN), nclusters])
            nclus_all.append(int(nclusters))    
        valB = int(np.mean(nclus_all[-5:]))
        #valB = nclus_all
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
            "g2" : [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4],
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
            "alpha" : []
            }
    

    eps = float(sys.argv[1])

    Bparams = {
            "avg_FR": [nE, nI, del_T], 
            "n_bumps": [nrowE, ncolE, eps, 10]        # [nrowE, ncolE, eps, min_samples]
            }

    #idA_params = Aparams[idA]
    idB_params = Bparams[idB]

    # Address for saving data
    saving_address = "/home/bhalla/shreyal/data/EI_brunel_{}/EI_serial_{}_{}_size{}_std{}-{}-{}-{}_J{}_g{}g2{}_nmE{}I{}_alpha{}_base{}_taum{}_pEE{}_A{}_B{}_{}trial_{}delT_eps{}/".format(asymmetry, asymmetry, landscape_type, landscape_size, stdEE, stdEI, stdIE, stdII, int(J), int(g), int(g2), int(noise_mean_E), int(noise_mean_I), alpha, seed, int(neuron_paramsE["tau_m"]), pEE, idA, idB, ntrials, del_T, eps)

    output_file1 = 'EI_spiketimes_' + asymmetry
    
    # running simulation - increasing 

    listB0_all = []
    
    if idA == 'input':
        noise_mean_E = listA[0]
        noise_mean_I = noise_mean_E
        noise = [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] 

    # CHECK SCRIPT AGAIN BEFORE RUNNING ---- *** SOME UNINTENDED CHANGES WERE MADE ***

    for nt in range(ntrials):

        #output_file = '{}{}_increasing_{}-18003-0'.format(saving_address, output_file1, nt)
        #simulation_parmeters = [t_warmup, del_T, noise, output_file]

        #ids0, ts0, listB0, timeAB0 = read_simulation_EI_serial(simulation_parmeters, [idA, listA, idB, idA_params, idB_params])
        #listB0_all.append(listB0)
        #del([ids0,ts0])

        output_file2 = '{}{}_increasing_{}_listB0_all.json'.format(saving_address, output_file1, nt)
        with open(output_file2, 'r') as write_file:
            #print(type(listB0))
            #json.dump(listB0, write_file)
            print("Increasing - {}".format(nt))
            listB0 = json.load(write_file)
        listB0_all.append(listB0)


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
         
        #output_file = '{}{}_decreasing_{}-18003-0'.format(saving_address, output_file1, nt)
        #simulation_parmeters = [t_warmup, del_T, noise, output_file]

        #ids1, ts1, listB1, timeAB1 = read_simulation_EI_serial(simulation_parmeters, [idA, listA, idB, idA_params, idB_params])
        #listB1_all.append(listB1)
        #del([ids1,ts1])

        output_file2 = '{}{}_decreasing_{}_listB1_all.json'.format(saving_address, output_file1, nt)
        with open(output_file2, 'r') as write_file:
            #json.dump(listB1, write_file)
            print("Decreasing - {}".format(nt))
            listB1 = json.load(write_file)
        listB1_all.append(listB1)


    listA1 = listA.copy()
    listB0_mean = np.mean(np.array(listB0_all), axis=0)
    listB1_mean = np.mean(np.array(listB1_all), axis=0)
    listB0_std = np.std(np.array(listB0_all), axis=0)
    listB1_std = np.std(np.array(listB1_all), axis=0)
    print([np.shape(listA1), np.shape(listB1_mean), np.shape(listB1_std)])

    # Plots

    """
    # Save the landscape   
    plt.imshow(landscape.reshape(nrowL,nrowL), origin='lower', cmap='twilight', vmin=0, vmax=8)
    plt.xlabel('Neuron grid x-axis', fontsize='x-large')
    plt.ylabel('Neuron grid y-axis', fontsize='x-large')
    cbar = plt.colorbar()
    cbar.set_label('Asymmetry directions', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('{}landscape'.format(saving_address))
    #plt.show()
   
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
    
    """
    # Plot increasing case
    fig, ax = plt.subplots()
    print(len(listB0_all), len(listB0_all[0]))
    A_len = len(listA0)
    B_len = len(listB0_all[0])

    new_listA0 = []
    for _t in range(A_len):
        new_listA0 += [listA0[_t]]*int(del_T/20.)
    time_all = np.arange(t_warmup, t_warmup + del_T*A_len, 20.)
    print([len(time_all, B_len)])
    ax.plot(time_all1, listA0, '-', color='#998ec3', label=idA)
    #ax.plot(time_all1, listA1, 'b--', label=idA)
    ax.set_ylabel(idA)
    ax.set_xlabel('Time')
    ax1 = ax.twinx()
    for i in range(ntrials):
        if i == 0:
            ax1.plot(time_all, listB0_all[i], '-', color='#f1a340', label=idB)
        else:
            ax1.plot(time_all, listB0_all[i], '-', color='#f1a340')
        #ax1.plot(time_all, listB1_all[i], 'r--', label=idB)
    ax1.set_ylabel(idB)
    plt.tight_layout()
    plt.savefig('{}brunel_serial_AB_time_multiple'.format(saving_address))
    """

    # Raster

    # Average firing rate
