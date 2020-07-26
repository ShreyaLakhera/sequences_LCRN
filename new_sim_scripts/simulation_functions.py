import numpy as np
import matplotlib.pyplot as plt
import nest
import glob
import lib.connectivity_landscape as cl
import lib.lcrn_network as lcrn
from scipy.spatial import distance
from scipy import sparse
from scipy.sparse import lil_matrix
import new_sim_scripts.test_sector_points_cartesian as dirconn
from scipy.interpolate import interp1d

def findFF(AM, N_start, n_cluster, nffn):
    """
    finds feed forward path given a starting set of neurons
    """
    
    F_all = [N_start]
    Fi = N_start
    N_FF = list(N_start)

    FF_full = -1
    #AM = lil_matrix.toarray(AM)

    spread_seq = []

    for i in range(nffn):
        AMi = np.zeros(np.shape(AM))
        AMi[Fi,:] = np.array(AM[Fi,:])
        AMi_sum = np.sum(AMi, axis=0)
        #print(np.unique(AMi_sum).astype(int))

        Pi = np.where(AMi_sum > 0.)[0]
        Pi = np.setdiff1d(Pi, N_FF)
        if np.size(Pi) < n_cluster:
            FF_full = 0
            print([i, 'broken'])
            break
        Fi1 = np.argsort(AMi_sum)[-n_cluster:]
        cd = np.mean(get_spread(Fi1, np.shape(AM)[0]))
        spread_seq.append(cd)
        FF_full = 1
        F_all.append(Fi1)
        N_FF += Fi1.tolist()
        Fi = Fi1

    plt.plot(spread_seq, '.')
    plt.show()

    # Calculate effective length

    nN = np.shape(AM)[0]
    nrow = ncol = int(np.sqrt(nN))

    F0 = F_all[0]
    F50 = F_all[-1]

    F0_coor = np.zeros((n_cluster, 2))
    F50_coor = np.zeros((n_cluster, 2))

    for i,n in enumerate(F0):
        F0_coor[i,0] = n // nrow        # x coor
        F0_coor[i,1] = n % nrow         # y coor
        
    for i,n in enumerate(F50):
        F50_coor[i,0] = n // nrow        # x coor
        F50_coor[i,1] = n % nrow         # y coor

    F0_centroid = (round(np.mean(F0_coor[:,0])), round(np.mean(F0_coor[:,1])))
    F50_centroid = (round(np.mean(F50_coor[:,0])), round(np.mean(F50_coor[:,1])))

    effective_length = float(distance.cdist([F0_centroid], [F50_centroid], 'euclidean'))
    
    return F_all, effective_length, FF_full

def findFF2(AM, iAM, N_start, n_cluster, nffn):
    """
    finds feed forward path given a starting set of neurons, neurons are chosen based on maximum excitation received and least inhibition received
    """
    
    F_all = [N_start]
    Fi = N_start
    N_FF = list(N_start)

    FF_full = -1
    #AM = lil_matrix.toarray(AM)
    #spread_seq = []

    for i in range(nffn):
        AMi = np.zeros(np.shape(AM))
        AMi[Fi,:] = np.array(AM[Fi,:])
        AMi_sum = np.sum(AMi, axis=0)
        
        iAMi = np.zeros(np.shape(iAM))
        iAMi[Fi,:] = np.array(iAM[Fi,:])
        iAMi_sum = np.sum(iAMi, axis=0)

        if i< 5:
            print(np.unique(iAMi_sum).astype(int))

        # --- 1
        Pi = np.arange(np.shape(AM)[0])
        Pi = np.setdiff1d(Pi, N_FF)

        # --- 2  
        #Pi = np.where(AMi_sum > 0.)[0]
        #Pi = np.setdiff1d(Pi, N_FF)

        # --- 3
        #Pi = np.setdiff1d(np.arange(np.shape(AM)[0]).astype(int), N_FF)

        if np.size(Pi) < n_cluster:
            FF_full = 0
            print([i, 'broken'])
            break

        # find the ones that have least inhibition
        
        # --- 1
        Fi1_id = np.argsort(AMi_sum[Pi] - iAMi_sum[Pi])[-n_cluster:]
        Fi1 = Pi[Fi1_id]
    
        # --- 2
        #Fi1_id = np.argsort(iAMi_sum[Pi])[:n_cluster]
        #Fi1 = Pi[Fi1_id]

        # --- 3
        #Fi1_id = np.argsort(AMi_sum[Pi] - iAMi_sum[Pi])[-n_cluster:]
        #Fi1 = Pi[Fi1_id]

        #print(AMi_sum[Fi1_id] - iAMi_sum[Fi1_id])

        #cd = np.max(get_spread(Fi1, np.shape(AM)[0]))
        #spread_seq.append(cd)

        FF_full = 1
        F_all.append(Fi1)
        N_FF += Fi1.tolist()
        Fi = Fi1
    
    #fig, ax = plt.subplots()
    #ax.plot(spread_seq, '.')
    #plt.show()

    # Calculate effective length

    nN = np.shape(AM)[0]
    nrow = ncol = int(np.sqrt(nN))

    F0 = F_all[0]
    F50 = F_all[-1]

    F0_coor = np.zeros((n_cluster, 2))
    F50_coor = np.zeros((n_cluster, 2))

    for i,n in enumerate(F0):
        F0_coor[i,0] = n // nrow        # x coor
        F0_coor[i,1] = n % nrow         # y coor
        
    for i,n in enumerate(F50):
        F50_coor[i,0] = n // nrow        # x coor
        F50_coor[i,1] = n % nrow         # y coor

    F0_centroid = (round(np.mean(F0_coor[:,0])), round(np.mean(F0_coor[:,1])))
    F50_centroid = (round(np.mean(F50_coor[:,0])), round(np.mean(F50_coor[:,1])))

    effective_length = float(distance.cdist([F0_centroid], [F50_centroid], 'euclidean'))
    
    return F_all, effective_length, FF_full

def find_dist_FF(AM, N_start, n_cluster, nffn):
    """
    finds distribution of number of connections when calculating FF path given a starting set of neurons
    """
    
    F_all = [N_start]
    Fi = N_start
    N_FF = list(N_start)

    FF_full = -1
    #AM = lil_matrix.toarray(AM)

    fig, ax = plt.subplots()
    ax.set_yscale("log")

    for i in range(nffn):
        AMi = np.zeros(np.shape(AM))
        AMi[Fi,:] = np.array(AM[Fi,:])
        AMi_sum = np.sum(AMi, axis=0)
        plt.hist(AMi_sum, bins=200)
        #print('\nZeros = {}'.format(len(np.where(AMi_sum == 0)[0])))
        #print('Ones = {}'.format(len(np.where(AMi_sum == 1)[0])))

        Pi = np.where(AMi_sum == 0.)[0]
        Pi = np.setdiff1d(Pi, N_FF)
        if np.size(Pi) < n_cluster:
            FF_full = 0
            print([i, 'broken'])
            break
        Fi1 = np.argsort(AMi_sum)[-n_cluster:]
        #Fi1 = np.random.choice(Pi, n_cluster, replace=False)
        FF_full = 1
        F_all.append(Fi1)
        N_FF += Fi1.tolist()
        Fi = Fi1

    plt.show()
    return F_all, FF_full

def get_coor(n):
    x = []
    y = []
    for i in n:
        ix = i // 120
        iy = i % 120
        x.append(ix)
        y.append(iy)
    return x,y

def get_spread(F, nN):
    x,y = get_coor(F.tolist())
    cx = np.mean(x)
    cy = np.mean(y)
    d = np.sqrt(np.square(np.array(x) - cx) + np.square(np.array(y) - cy))
    #print(d)
    return d

def chooseF0(nN, n_cluster, prev_F0_centroids):

    sample_set = np.setdiff1d(np.arange(nN), np.array(prev_F0_centroids))
    F0_centroid = np.random.choice(sample_set)

    nrow = ncol = int(np.sqrt(nN))

    x_F0c = F0_centroid // nrow
    y_F0c = F0_centroid % nrow

    r = int(np.sqrt(n_cluster)/2)

    N_start = []

    for x in range(x_F0c-r, x_F0c+r):
        for y in range(y_F0c-r, y_F0c+r):
            n = ((x % nrow)*nrow + (y % ncol))
            N_start.append(n)

    return F0_centroid, N_start

def chooseF0_n(nN, F0_centroid, n_cluster):

    nrow = ncol = int(np.sqrt(nN))

    x_F0c = F0_centroid // nrow
    y_F0c = F0_centroid % nrow

    r = int(np.sqrt(n_cluster)/2)

    N_start = []

    for x in range(x_F0c-r, x_F0c+r):
        for y in range(y_F0c-r, y_F0c+r):
            n = ((x % nrow)*nrow + (y % ncol))
            N_start.append(n)

    return F0_centroid, N_start



def run_simulation(pop_all, simulation_parameters):
    
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, noise_mean, noise_std, output_file] = simulation_parameters

    noise_params = {'mean': noise_mean, 'std': noise_std}

    noiseE = nest.Create('noise_generator', params=noise_params)
    noiseI = nest.Create('noise_generator', params=noise_params)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)

    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def run_simulation_EI(pop_all, simulation_parameters):
    
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, noise, output_file] = simulation_parameters

    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I] = noise

    noise_params_E = {'mean': noise_mean_E, 'std': noise_std_E}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}

    noiseE = nest.Create('noise_generator', params=noise_params_E)
    noiseI = nest.Create('noise_generator', params=noise_params_I)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(noiseE, popE)
    nest.Connect(noiseI, popI)

    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    del(pop_all, pop, popE, popI, noise_params_E, noise_params_I, noiseE, noiseI, sd, sd_stat, noise, output_file)

    return ids, ts

def run_simulation_1seq(pop_all, simulation_parameters, n_cutoff):
    
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, noise, output_file] = simulation_parameters

    [noise_mean_E1, noise_std_E1, noise_mean_E2, noise_std_E2, noise_mean_I, noise_std_I] = noise

    noise_params_E1 = {'mean': noise_mean_E1, 'std': noise_std_E1}
    noise_params_E2 = {'mean': noise_mean_E2, 'std': noise_std_E2}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}

    noiseE1 = nest.Create('noise_generator', params=noise_params_E1)
    noiseE2 = nest.Create('noise_generator', params=noise_params_E2)
    noiseI = nest.Create('noise_generator', params=noise_params_I)

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    popE1 = popE[:n_cutoff]
    popE2 = popE[n_cutoff:] 

    nest.Connect(noiseE1, popE1)
    nest.Connect(noiseE2, popE2)
    nest.Connect(noiseI, popI)

    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def run_simulation_EI_ipg(pop_all, simulation_parameters):
    """
    Run simulation with inhomogenous poisson generator as input to the netowork
    """
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, pg_input, output_file] = simulation_parameters
    [timesE_pg, valuesE_pg, timesI_pg, valuesI_pg] = pg_input

    print([np.shape(timesE_pg), np.shape(valuesE_pg)])
    print([np.shape(timesI_pg), np.shape(valuesI_pg)])

    ipgE = nest.Create("inhomogeneous_poisson_generator", params={'rate_times':timesE_pg, 'rate_values':valuesE_pg})
    ipgI = nest.Create("inhomogeneous_poisson_generator", params={'rate_times':timesI_pg, 'rate_values':valuesI_pg})

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(ipgE, popE)
    nest.Connect(ipgI, popI)

    nest.Connect(pop, sd)

    nest.Simulate(t_warmup)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def run_simulation_EI_ipg_1seq(pop_all, simulation_parameters, n_cutoff):
    
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, pg_input, output_file, iN, w] = simulation_parameters
    [timesE, valuesE, timesI, valuesI, timesP, valuesP] = pg_input

    ipgE = nest.Create("inhomogeneous_poisson_generator", params={'rate_times':timesE, 'rate_values':valuesE})
    ipgI = nest.Create("inhomogeneous_poisson_generator", params={'rate_times':timesI, 'rate_values':valuesI})
    ipgP = nest.Create("inhomogeneous_poisson_generator", params={'rate_times':timesP, 'rate_values':valuesP})
    
    # Get neurons in area of directed simulation
    ncA = []
    ncN_x = int(inp_center_N[0])
    ncN_y = int(inp_center_N[1])
    
    w = int(width/2)

    for i in range(ncN_x - w,ncN_x + w):
        for j in range(ncN_y - w,ncN_y + w):
            n = ((j * ncolE) + i) % nE
            ncA.append(popE[n])

    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(ipgE, popE)
    nest.Connect(ipgI, popI)
    nest.Connect(pop, sd)
    nest.Simulate(t_warmup)

    nest.Connect(ipgP, ncA)
    nest.Simulate(t_sim)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts


def get_grid_around(inpN,w, nN, pop):
    [x0,y0] = inpN
    if nN < 14300:
        [x0, y0] = [int(x0/2), int(y0/2)]
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

def run_simulation_dynamic_input(pop_all, simulation_parameters, neuron_parameters):	

    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [pop, popE, popI] = pop_all
    [t_warmup, t_sim, t_step, noise_input_params, output_file, inp_center_N, w] = simulation_parameters
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

    n_steps = int(np.floor(t_sim/t_step))    

    for i in range(n_steps):
        ncA = get_grid_around(inp_center_N, w, nE, popE)
        sp_noise = nest.Create("noise_generator", params = noise_params_pulse)
        nest.Connect(sp_noise, ncA)
        nest.Simulate(t_step)
        reset_params = {'mean':0., 'std':0.}
        nest.SetStatus(sp_noise, params=reset_params)

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts

def run_simulation_EI_directed_input(pop_all, simulation_parameters, neuron_parameters):
    
    [pop, popE, popI] = pop_all
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [t_warmup, t_burst, t_sim, noise, output_file, inp_center_N, width] = simulation_parameters
    [noise_mean_E, noise_std_E, noise_mean_I, noise_std_I, noise_mean_pulse, noise_std_pulse] = noise
    
    # Get neurons in area of directed simulation
    ncA = []
    ncN_x = int(inp_center_N[0])
    ncN_y = int(inp_center_N[1])
    
    w = int(width/2)

    for i in range(ncN_x - w,ncN_x + w):
        for j in range(ncN_y - w,ncN_y + w):
            n = ((j * ncolE) + i) % nE
            ncA.append(popE[n])


    # set different noise input values
    noise_params_E = {'mean': noise_mean_E, 'std': noise_std_E}
    noise_params_I = {'mean': noise_mean_I, 'std': noise_std_I}
    noise_params_pulse = {'mean': noise_mean_pulse, 'std': noise_std_pulse}
    reset_params = {'mean': 0., 'std': 0.}

    # Create the different noise devices
    noiseE = nest.Create('noise_generator', params=noise_params_E)              # background input to E
    noiseI = nest.Create('noise_generator', params=noise_params_I)              # background input to I
    sp_noise = nest.Create("noise_generator", params = noise_params_pulse)      # directed input

    # Create spike detecter device
    sd = nest.Create('spike_detector', params={'to_file':True, 'label':output_file})

    nest.Connect(noiseE, popE)          # connect background noise to E
    nest.Connect(noiseI, popI)          # connect background noise to I
    nest.Connect(pop, sd)               # connect neurons to spike detector

    nest.Simulate(t_warmup)             # warmup simulation: background input

    nest.Connect(sp_noise, ncA)         # connect the directed input to specific area ncA
    nest.Simulate(t_burst)              # main simulation: background + directed input
 
    nest.SetStatus(sp_noise, params=reset_params)   # set directed input = 0
    nest.SetStatus(noiseE, params=reset_params)     # set background input = 0
    nest.SetStatus(noiseI, params=reset_params)     # set background input = 0
    nest.Simulate(t_sim)                # control simulation: 0 input to all

    sd_stat = nest.GetStatus(sd, 'events')[0]
    ids = sd_stat['senders']
    ts = sd_stat['times']

    return ids, ts


def create_connectivity_EI(neuron_parameters, connectivity_parameters, save_AM_parameters):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [landscape_type, landscape_size, asymmetry, p, std, shift, seed] = connectivity_parameters
    [pEE, pEI, pIE, pII] = p
    [stdEE, stdEI, stdIE, stdII] = std
    [AM_address, fname] = save_AM_parameters

    if asymmetry[0] == 'E':
        nrowL = nrowE
    else:
        nrowL = nrowI

    if asymmetry[-1] == 'E':
        nrowM = nrowE
    else:
        nrowM = nrowI

    move = cl.move(nrowM)
    
    if landscape_type == 'symmetric':
        landscape = None                                        
    elif landscape_type == 'random':
        landscape = cl.random(nrowL, {'seed': 0})
    elif landscape_type == 'homogenous':
        landscape = cl.homogeneous(nrowL, {'phi': 3})
    elif landscape_type == 'perlin':
        landscape = cl.Perlin(nrowL, {'size': int(landscape_size)})
    elif landscape_type == 'perlinuniform':
        landscape = cl.Perlin_uniform(nrowL, {'size': int(landscape_size), 'base': seed})

    AM = np.zeros((nN, nN))

    for idx in range(nE):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowE, ncolE, int(pEE * nE), stdEE)
        if asymmetry == 'EE':
            if landscape is not None:
                targets = (targets + shift * move[landscape[idx] % len(move)]) % nE
        targets = targets[targets != idx].astype(int)
        AM[idx, targets] = 1.

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowI, ncolI, int(pEI * nI), stdEI)
        if asymmetry == 'EI':
            if landscape is not None:
                targets = ((targets + shift * move[landscape[idx] % len(move)]) % nI).astype(int)
        AM[idx, targets + nE] = 1. 

    for idx in range(nI):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowE, ncolE, int(pIE * nE), stdIE)
        if asymmetry == 'IE':
            if landscape is not None:
                targets = ((targets + shift * move[landscape[idx] % len(move)]) % nE).astype(int)
        AM[idx + nE, targets] = 1.

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowI, ncolI, int(pII * nI), stdII)
        if asymmetry == 'II':
            if landscape is not None:
                targets = (targets + shift * move[landscape[idx] % len(move)]) % nI
        targets = targets[targets != idx].astype(int)
        AM[idx + nE, targets + nE] = 1.
    
    sparse.save_npz(AM_address + fname + 'normalAM', sparse.coo_matrix(AM))
    sparse.save_npz(AM_address + fname + 'landscape', sparse.coo_matrix(landscape))
 
    return [AM, landscape, nrowL]


def create_connectivity_EI_brunel_old_asymm(neuron_parameters, connectivity_parameters, save_AM_parameters):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [landscape_type, landscape_size, asymmetry, p, shift, std, alpha, seed] = connectivity_parameters
    [pEE, pEI, pIE, pII] = p
    [stdEE, stdEI, stdIE, stdII] = std
    [AM_address, fname] = save_AM_parameters
    [alphaEE, alphaEI, alphaIE, alphaII] = [alpha, alpha, alpha, alpha]

    if asymmetry[0] == 'E':
        nrowL = nrowE
    else:
        nrowL = nrowI

    if asymmetry[-1] == 'E':
        nrowM = nrowE
    else:
        nrowM = nrowI

    move = cl.move(nrowM)
    
    if landscape_type == 'symmetric':
        landscape = None                                        
    elif landscape_type == 'random':
        landscape = cl.random(nrowL, {'seed': 0})
    elif landscape_type == 'homogenous':
        landscape = cl.homogeneous(nrowL, {'phi': 3})
    elif landscape_type == 'perlin':
        landscape = cl.Perlin(nrowL, {'size': int(landscape_size)})
    elif landscape_type == 'perlinuniform':
        landscape = cl.Perlin_uniform(nrowL, {'size': int(landscape_size), 'base': seed})

    AM = np.zeros((nN, nN))
    ran_AM = np.zeros((nN, nN))

    for idx in range(nE):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowE, ncolE, int(alphaEE * pEE * nE), stdEE)
        if (asymmetry == 'EE') and (landscape is not None):
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nE
        targets = targets[targets != idx].astype(int)
        AM[idx, targets] = 1.
        r_targets = get_random_targets(idx, nrowE, ncolE, nrowE, ncolE, int(pEE * nE * (1 - alphaEE)), targets)
        ran_AM[idx, r_targets] = 1.

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowI, ncolI, int(alphaEI * pEI * nI), stdEI)
        if (asymmetry == 'EI') and (landscape is not None):
            targets = ((targets + shift * move[landscape[idx] % len(move)]) % nI).astype(int)
        AM[idx, targets + nE] = 1. 
        r_targets = get_random_targets(idx, nrowE, ncolE, nrowI, ncolI, int(pEI * nI * (1 - alphaEI)), targets)
        ran_AM[idx, r_targets + nE] = 1.

    for idx in range(nI):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowE, ncolE, int(alphaIE * pIE * nE), stdIE)
        if (asymmetry == 'IE') and (landscape is not None):
            targets = ((targets + shift * move[landscape[idx] % len(move)]) % nE).astype(int)
        AM[idx + nE, targets] = 1.
        r_targets = get_random_targets(idx, nrowI, ncolI, nrowE, ncolE, int(pIE * nE * (1 - alphaIE)), targets)
        ran_AM[idx + nE, r_targets] = 1.

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowI, ncolI, int(alphaII * pII * nI), stdII)
        if (asymmetry == 'II') and (landscape is not None):
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nI
        targets = targets[targets != idx].astype(int)
        AM[idx + nE, targets + nE] = 1.
        r_targets = get_random_targets(idx, nrowI, ncolI, nrowI, ncolI, int(pII * nI * (1 - alphaII)), targets)
        ran_AM[idx + nE, r_targets + nE] = 1.
    
    print('Multiple connections')
    print(np.where(ran_AM+AM > 1.))

    sparse.save_npz(AM_address + fname + 'brunel_old_asymm_rAM', sparse.coo_matrix(ran_AM))
    sparse.save_npz(AM_address + fname + 'brunel_old_asymm_dAM', sparse.coo_matrix(AM))
    sparse.save_npz(AM_address + fname + 'brunel_old_asymm_landscape', sparse.coo_matrix(landscape))
 
    return [ran_AM, AM, landscape, nrowL]
 

def create_connectivity_EI_dir(neuron_parameters, connectivity_parameters, save_AM_parameters):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [landscape_type, landscape_size, asymmetry, p, std, separation, width, alpha, seed] = connectivity_parameters
    [pEE, pEI, pIE, pII] = p
    [stdEE, stdEI, stdIE, stdII] = std
    [AM_address, fname] = save_AM_parameters

    if asymmetry[0] == 'E':
        nrowL = nrowE
    else:
        nrowL = nrowI
 
    if landscape_type == 'symmetric':
        landscape = None                                        
    elif landscape_type == 'random':
        landscape = cl.random(nrowL, {'seed': 0})
    elif landscape_type == 'homogenous':
        landscape = cl.homogeneous(nrowL, {'phi': 3})
    elif landscape_type == 'perlin':
        landscape = cl.Perlin(nrowL, {'size': int(landscape_size)})
    elif landscape_type == 'perlinuniform':
        landscape = cl.Perlin_uniform(nrowL, {'size': int(landscape_size), 'base': seed})

    iso_AM = np.zeros((nN, nN))
    dir_AM = np.zeros((nN, nN))

    [alphaEE, alphaEI, alphaIE, alphaII] = [0,0,0,0]
    if asymmetry == 'EE':
        alphaEE = alpha
    elif asymmetry == 'EI':
        alphaEI = alpha
    elif asymmetry == 'IE':
        alphaIE = alpha
    elif asymmetry == 'II':
        alphaII = alpha

    for idx in range(nE):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowE, ncolE, int(pEE * nE * (1 - alphaEE)), stdEE)
        targets = targets[targets != idx]
        if asymmetry == 'EE':
            if landscape is not None:
                direction = landscape[idx]
                dir_targets = dirconn.get_directional_targets(idx, nrowE, ncolE, nrowE, ncolE, direction, separation, width, int(pEE * nE * alphaEE))
                dir_targets = dir_targets[dir_targets != idx]
                targets = np.setdiff1d(targets, dir_targets)
                dir_AM[idx, dir_targets] = 1.
        iso_AM[idx, targets] = 1. 

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowI, ncolI, int(pEI * nI * (1 - alphaEI)), stdEI)
        if asymmetry == 'EI':
            if landscape is not None:
                direction = landscape[idx]
                dir_targets = dirconn.get_directional_targets(idx, nrowE, ncolE, nrowI, ncolI, direction, separation, width, int(pEI * nI * alphaEI))
                targets = np.setdiff1d(targets, dir_targets)
                dir_AM[idx, dir_targets + nE] = 1.
        iso_AM[idx, targets + nE] = 1.

    for idx in range(nI):
        
        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowE, ncolE, int(pIE * nE * (1 - alphaIE)), stdIE)
        if asymmetry == 'IE':
            if landscape is not None:
                direction = landscape[idx]
                dir_targets = dirconn.get_directional_targets(idx, nrowI, ncolI, nrowE, ncolE, direction, separation, width, int(pIE * nE * alphaIE))
                targets = np.setdiff1d(targets, dir_targets)
                dir_AM[idx + nE, dir_targets] = 1.
        iso_AM[idx + nE, targets] = 1.

        targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowI, ncolI, int(pII * nI * (1 - alphaII)), stdII)
        targets = targets[targets != idx]
        if asymmetry == 'II':
            if landscape is not None:
                direction = landscape[idx]
                dir_targets = dirconn.get_directional_targets(idx, nrowI, ncolI, nrowI, ncolI, direction, separation, width, int(pII * nI * alphaII))
                dir_targets = dir_targets[dir_targets != idx]
                targets = np.setdiff1d(targets, dir_targets)
                dir_AM[idx + nE, dir_targets + nE] = 1.
        iso_AM[idx + nE, targets + nE] = 1.

    print('Multiple connections')
    print(np.where(iso_AM+dir_AM > 1.))

    sparse.save_npz(AM_address + fname + 'isoAM', sparse.coo_matrix(iso_AM))
    sparse.save_npz(AM_address + fname + 'dirAM', sparse.coo_matrix(dir_AM))
    sparse.save_npz(AM_address + fname + 'landscape', sparse.coo_matrix(landscape))
 
    return [iso_AM, dir_AM, landscape, nrowL]

def get_random_targets(idx, nsrow, nscol, ntrow, ntcol, nconn, ignore_set):  
    sample_set = np.setdiff1d(np.arange(ntrow*ntcol), ignore_set)
    sample_set = np.setdiff1d(sample_set, np.array([idx]))
    targets = np.random.choice(sample_set, nconn)
    if nsrow == ntrow:
        targets = targets[targets != idx]
    return targets


def create_connectivity_EI_random_dir(neuron_parameters, connectivity_parameters, save_AM_parameters):
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [landscape_type, landscape_size, asymmetry, p, shift, std, alpha, seed] = connectivity_parameters
    [pEE, pEI, pIE, pII] = p
    [stdEE, stdEI, stdIE, stdII] = std
    [AM_address, fname] = save_AM_parameters

    if asymmetry[0] == 'E':
        nrowL = nrowE
    else:
        nrowL = nrowI
    
    if asymmetry[-1] == 'E':
        nrowM = nrowE
    else:
        nrowM = nrowI
    move = cl.move(nrowM)
 
    if landscape_type == 'symmetric':
        landscape = None                                        
    elif landscape_type == 'random':
        landscape = cl.random(nrowL, {'seed': 0})
    elif landscape_type == 'homogenous':
        landscape = cl.homogeneous(nrowL, {'phi': 3})
    elif landscape_type == 'perlin':
        landscape = cl.Perlin(nrowL, {'size': int(landscape_size)})
    elif landscape_type == 'perlinuniform':
        landscape = cl.Perlin_uniform(nrowL, {'size': int(landscape_size), 'base': seed})

    ran_AM = np.zeros((nN, nN))
    dir_AM = np.zeros((nN, nN))

    [alphaEE, alphaEI, alphaIE, alphaII] = [0,0,0,0]
    if asymmetry == 'EE':
        alphaEE = alpha
    elif asymmetry == 'EI':
        alphaEI = alpha
    elif asymmetry == 'IE':
        alphaIE = alpha
    elif asymmetry == 'II':
        alphaII = alpha

    for idx in range(nE):
        
        targets = []
        if (asymmetry == 'EE') and (landscape is not None):
            targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowE, ncolE, int(pEE * nE * alphaEE), stdEE)
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nE
            targets = targets[targets != idx].astype(int)
            dir_AM[idx, targets] = 1.
        r_targets = get_random_targets(idx, nrowE, ncolE, nrowE, ncolE, int(pEE * nE * (1 - alphaEE)))
        r_targets = np.setdiff1d(r_targets, targets)
        ran_AM[idx, r_targets] = 1.
        
        targets = []
        if (asymmetry == 'EI') and (landscape is not None):
            targets, delays = lcrn.lcrn_gauss_targets(idx, nrowE, ncolE, nrowI, ncolI, int(pEI * nI * alphaEI), stdEI)
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nI
            dir_AM[idx, targets + nE] = 1.
        r_targets = get_random_targets(idx, nrowE, ncolE, nrowI, ncolI, int(pEI * nI * (1 - alphaEI)))
        r_targets = np.setdiff1d(r_targets, targets)
        ran_AM[idx, r_targets + nE] = 1.

    for idx in range(nI):
        
        targets = []
        if (asymmetry == 'IE') and (landscape is not None):
            targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowE, ncolE, int(pIE * nE * alphaIE), stdIE)
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nI
            dir_AM[idx + nE, targets] = 1.
        r_targets = get_random_targets(idx, nrowI, ncolI, nrowE, ncolE, int(pIE * nE * (1 - alphaIE)))
        r_targets = np.setdiff1d(r_targets, targets)
        ran_AM[idx + nE, r_targets] = 1.

        targets = []
        if (asymmetry == 'II') and (landscape is not None):
            targets, delays = lcrn.lcrn_gauss_targets(idx, nrowI, ncolI, nrowI, ncolI, int(pII * nI * alphaII), stdII)
            targets = (targets + shift * move[landscape[idx] % len(move)]) % nI
            targets = targets[targets != idx].astype(int)
            dir_AM[idx + nE, targets + nE] = 1.
        r_targets = get_random_targets(idx, nrowI, ncolI, nrowI, ncolI, int(pII * nI * (1 - alphaII)))
        r_targets = np.setdiff1d(r_targets, targets)
        ran_AM[idx + nE, r_targets + nE] = 1.

    print('Multiple connections')
    print(np.where(ran_AM+dir_AM > 1.))

    sparse.save_npz(AM_address + fname + 'random_rAM', sparse.coo_matrix(ran_AM))
    sparse.save_npz(AM_address + fname + 'random_dAM', sparse.coo_matrix(dir_AM))
    sparse.save_npz(AM_address + fname + 'landscape', sparse.coo_matrix(landscape))
 
    return [ran_AM, dir_AM, landscape, nrowL]

def read_AM_dir_sparse(save_AM_parameters):
    [AM_address, fname] = save_AM_parameters

    isoAM = sparse.load_npz(AM_address + fname + 'isoAM.npz').toarray()
    dirAM = sparse.load_npz(AM_address + fname + 'dirAM.npz').toarray()
    landscape = sparse.load_npz(AM_address + fname + 'landscape.npz').toarray()
    nrowL = np.sqrt(int(np.size(landscape)))
     
    return [isoAM, dirAM, landscape, int(nrowL)]

def read_AM_random_dir_sparse(save_AM_parameters):
    [AM_address, fname] = save_AM_parameters

    isoAM = sparse.load_npz(AM_address + fname + 'random_rAM.npz').toarray()
    dirAM = sparse.load_npz(AM_address + fname + 'random_dAM.npz').toarray()
    landscape = sparse.load_npz(AM_address + fname + 'landscape.npz').toarray()
    nrowL = np.sqrt(int(np.size(landscape)))
     
    return [isoAM, dirAM, landscape, int(nrowL)]

def read_AM_brunel_old_asymm_sparse(save_AM_parameters):
    [AM_address, fname] = save_AM_parameters

    isoAM = sparse.load_npz(AM_address + fname + 'brunel_old_asymm_rAM.npz').toarray()
    dirAM = sparse.load_npz(AM_address + fname + 'brunel_old_asymm_dAM.npz').toarray()
    landscape = sparse.load_npz(AM_address + fname + 'brunel_old_asymm_landscape.npz').toarray()
    nrowL = np.sqrt(int(np.size(landscape)))
     
    return [isoAM, dirAM, landscape, int(nrowL)]

def read_AM_normal_sparse(save_AM_parameters):
    [AM_address, fname] = save_AM_parameters

    AM = sparse.load_npz(AM_address + fname + 'normalAM.npz').toarray()
    landscape = sparse.load_npz(AM_address + fname + 'landscape.npz').toarray()
    nrowL = np.sqrt(int(np.size(landscape)))
     
    return [AM, landscape, int(nrowL)]

def connect_network(neuron_parameters, synapse_parameters, AM, pop_all):

    [pop, popE, popI] = pop_all
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [Je, g] = synapse_parameters 

    offsetE = popE[0]
    offsetI = popI[0]

    syn_specE = {'weight': Je}
    syn_specI = {'weight': g * -Je}

    for idx in range(nE):
        targets = np.nonzero(AM[idx,:])[0] + offsetE
        nest.Connect([popE[idx]], targets.tolist(), syn_spec=syn_specE)

    for idx in range(nI):
        targets = np.nonzero(AM[idx+nE,:])[0] + offsetE
        nest.Connect([popI[idx]], targets.tolist(), syn_spec=syn_specI)
    
    del([syn_specE, syn_specI, targets])

    return

def create_network(neuron_parameters, synapse_parameters, AM):

    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [Je, g] = synapse_parameters 
        
    popE = nest.Create(neuron_type, nE, params=neuron_paramsE)
    popI = nest.Create(neuron_type, nI, params=neuron_paramsI)

    pop = popE + popI
    pop_all = [pop, popE, popI]

    connect_network(neuron_parameters, synapse_parameters, AM, pop_all)

    return pop_all

def connect_network_directions(neuron_parameters, synapse_parameters, AM, pop_all):
    [pop, popE, popI] = pop_all
    [nrowE, ncolE, nrowI, ncolI, nE, nI, nN, neuron_type, neuron_paramsE, neuron_paramsI] = neuron_parameters
    [J, g, g2] = synapse_parameters
    [iso_AM, dir_AM] = AM 
	
    offsetE = popE[0]
    offsetI = popI[0]

    for idx in range(nE):
        iso_targets = np.where(iso_AM[idx,:] == 1.)[0] + offsetE
        dir_targets = np.where(dir_AM[idx,:] == 1.)[0] + offsetE
        nest.Connect([popE[idx]], iso_targets.tolist(), syn_spec={'weight': J})
        nest.Connect([popE[idx]], dir_targets.tolist(), syn_spec={'weight': g2 * J})
        del([iso_targets, dir_targets])

    for idx in range(nI):
        iso_targets = np.where(iso_AM[idx + nE,:] == 1.)[0] + offsetE
        dir_targets = np.where(dir_AM[idx + nE,:] == 1.)[0] + offsetE
        nest.Connect([popI[idx]], iso_targets.tolist(), syn_spec={'weight': -g * J})
        nest.Connect([popI[idx]], dir_targets.tolist(), syn_spec={'weight': -g * g2 * J})
        del([iso_targets, dir_targets])

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

def findFF_synlist(AM, N_start, n_cluster, nffn):
    """
    finds feed forward path given a starting set of neurons
    """
    
    F_all = [N_start]
    Fi = N_start
    N_FF = list(N_start)

    FF_full = -1

    for i in range(nffn):
        AMi = np.zeros(np.shape(AM))
        AMi[Fi,:] = np.array(AM[Fi,:])
        AMi_sum = np.sum(AMi, axis=0)
        #print(np.unique(AMi_sum).astype(int))

        Pi = np.where(AMi_sum > 0.)[0]
        Pi = np.setdiff1d(Pi, N_FF)
        if np.size(Pi) < n_cluster:
            FF_full = 0
            print([i, 'broken'])
            break
        Fi1 = np.argsort(AMi_sum)[-n_cluster:]
        FF_full = 1
        F_all.append(Fi1)
        N_FF += Fi1.tolist()

        Fi = Fi1

    return F_all, FF_full

def get_interpolated_curve(nrowE, ncolE, nsteps):
    x = np.random.choice(np.arange(0, nrowE, 1), size=int(nsteps/2), replace=False)
    y = np.random.choice(np.arange(0, ncolE, 1), size=int(nsteps/2), replace=False)
    ix = np.argsort(x)
    print(x, y)
    f = interp1d(x[ix], y[ix], kind='cubic')
    xn = np.arange(x.min(), x.max(), (x.max() - x.min())/nsteps)
    yn = f(xn) % ncolE
    return xn.astype(int), yn.astype(int)
