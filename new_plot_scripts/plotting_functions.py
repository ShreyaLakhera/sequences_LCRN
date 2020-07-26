import numpy as np
import matplotlib.pyplot as plt
#import lib.animation as anim
import glob
import pylab as pl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import lib.animation as anim
import scipy.sparse as sparse

def get_spiketimes(address, nfiles):
    if address[0] == '~':
        address = '/home/shreya' + address[1:]                  

    filenames = glob.glob(address + '*.gdf') 

    if len(filenames) != int(nfiles):
        print('Number of files does not match!')

    gids, ts = np.loadtxt(filenames[0]).T

    for _f in filenames[1:]:
        _gids, _ts = np.loadtxt(_f).T
        gids = np.hstack((gids, _gids))
        #_ts += ts[-1]
        ts = np.hstack((ts, _ts))

    return ts, gids

def get_spiketimes2(address, nfiles):
    if address[0] == '~':
        address = '/home/shreya' + address[1:]                  

    filenames = glob.glob(address + '.gdf') 

    if len(filenames) != int(nfiles):
        print('Number of files does not match!')

    gids, ts = np.loadtxt(filenames[0]).T

    for _f in filenames[1:]:
        _gids, _ts = np.loadtxt(_f).T
        gids = np.hstack((gids, _gids))
        #_ts += ts[-1]
        ts = np.hstack((ts, _ts))

    return ts, gids


def calculate_firing_rate(ts, gids, nN, tsim):
    firing_rate, ed = np.histogram(gids, bins=range(nN+1))
    firing_rate = firing_rate/tsim
    return firing_rate # Array

def get_FR_time(ts, gids, nN, t_step, t_start, t_end):
    ts_bins = np.arange(t_start, t_end + 1, t_step)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    FR = h.T
    return(FR)

def sortFR(FR):
    """
    sort neurons acc to their peak of FR
    gives back sorted FR and sort order
    """
    max_fr_time = np.argmax(FR, axis=1)
    sort_id = np.argsort(max_fr_time)
    sorted_FR = FR[sort_id,:]

    return sorted_FR, sort_id

def make_animation(ts, gids, nrow, ncol, nN, tres):
    simtime = np.max(ts)

    ts_bins = np.arange(0., simtime + 1, tres)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    fig, ax = pl.subplots(1)
    a = anim.images(ax, hh, vmin=0, vmax=np.max(hh))
    return a

def getSample(ts, ids, nN, nSN, spike_threshold):
    
    ts_sel = []
    ids_sel = []
    selected_N = []

    while len(selected_N) < nSN:
        sample_set = np.setdiff1d(np.arange(nN), np.array(selected_N))
        new_N = np.random.choice(sample_set, nSN-len(selected_N), replace=False)
        
        for i in new_N:
            idx = np.where(ids == i)[0]
            ts_new = ts[idx]
            ids_new = ids[idx]

            if len(idx) >= spike_threshold:
                selected_N += [i]
                new_Nid = len(selected_N) - 1
                ts_sel += ts_new.tolist()
                ids_sel += [new_Nid]*len(idx)

            if len(selected_N) == nSN:
                break

    return np.array(ts_sel), np.array(ids_sel), selected_N

def getSpiketimesGivenNeurons(ts, ids, selected_N):
    
    ts_sel = []
    ids_sel = []

    for i in selected_N:
        idx = np.where(ids == i)[0]
        ts_new = ts[idx]
        ids_new = ids[idx]

        ts_sel += ts_new.tolist()
        ids_sel += [i]*len(idx)

    return np.array(ts_sel), np.array(ids_sel)


def getSpikesTimeInterval(ts, ids, t_start, t_end):

    idt1 = ts > t_start
    idt2 = ts <= t_end

    idt = idt1 * idt2

    ts_interval = ts[idt]
    ids_interval = ids[idt]

    return ts_interval, ids_interval

def reassignNeuronID(ts, ids, nids1, nids2):
    """
    Given two sets of neurons ids (which is a mapping from 1 to another), it makes the permutation and gives back only the spiketimes of these
    neurons
    """

    if len(nids1) != len(nids2):
        print("Mapping between different number of neurons! Check nids1, nids2")

    ts_new = []
    ids_new = []

    for i,N in enumerate(nids1):
        idt = np.where(ids == N)[0]
        ts_new += ts[idt].tolist()
        ids_new += [nids2[i]]*len(idt)

    return ts_new, ids_new

def arrangeNeuronsEmptyGrid(selected_N, nrow, ncol, nN):
    """
    This gives the mapping of sample neurons that are sorted to the n_ids where they will be arranged on the final grid
    """

    n_samples = len(selected_N)

    x_level = 50
    grid_res = ncol // n_samples

    new_ids = np.arange(x_level*nrow, (x_level*nrow + n_samples*grid_res), grid_res)

    return new_ids

def calculateFiringInterval(ts, ids, nSN):

    firing_interval = np.zeros(nSN)
    t_midfiringinterval = np.zeros(nSN)
    t_startfiring = np.zeros(nSN)

    ts = np.array(ts)
    ids = np.array(ids)

    for i in range(nSN):
        print(i)
        id_N = np.where(ids == i)[0]
        ts_N = ts[id_N]
        start_times = []
        end_times = []
        mid_times = []
        
        _st = -1
        _lt = -1
        _et = -1
    
        for _t in ts_N:
            
            if (_lt != -1) and (_t - _lt > 50):
                _et = _lt
                start_times.append(_st)
                end_times.append(_et)
                mid_times.append(np.mean([_st,_et]))

            if (_t - _lt > 50):                
                _st = _t

            _lt = _t   

        print(start_times)
        print(end_times)

        all_intervals = np.array(end_times) - np.array(start_times)
        interval = np.mean(all_intervals)
        firing_interval[i] = interval
        std_FI = np.std(all_intervals)

        mid_times = np.array(mid_times)
        period = np.mean(mid_times[1:] - mid_times[:-1])
        t_midfiringinterval[i] = period
        std_t_mfi = np.std(mid_times[1:] - mid_times[:-1])

        start_times = np.array(start_times)
        period = np.mean(start_times[1:] - start_times[:-1])
        t_startfiring[i] = period
        std_t_sf = np.std(start_times[1:] - start_times[:-1])

        print("Std dev in firing interval for N {} = {}\nStd dev in t_midFI = {}\nStd dev in t_sf = {}\n".format(i, std_FI, std_t_mfi, std_t_sf))

    print(firing_interval)

    return firing_interval, t_midfiringinterval, t_startfiring # mean firing interval for each neuron

def get_quiver_data(landscape, nrowL):

    if nrowL == 120:
        nstep = 8
    else:
        nstep = 4

    x = np.arange(0,120,8)
    y = np.arange(0,120,8)
    #x = np.arange(8,128,8)
    #y = np.arange(8,128,8)

    X, Y = np.meshgrid(x, y)

    landscape1 = np.reshape(landscape, (nrowL, nrowL))
    #landscape1 = landscape1[np.arange(0,nrowL,nrowL/15).astype(int),:][:,np.arange(0,nrowL,nrowL/15).astype(int)]
    landscape1 = landscape1[np.arange(0,nrowL,nstep).astype(int),:][:,np.arange(0,nrowL,nstep).astype(int)]

    x_move = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    y_move = np.array([0, 1, 1, 1, 0, -1, -1, -1])
   
    u = x_move[landscape1]
    v = y_move[landscape1]

    return [X, Y, u, v]

def draw_landscape_quiver(d, ax):
    [X, Y, u, v] = d

    #fig, ax = plt.subplots()
    q = ax.quiver(X, Y, u, v, pivot='mid')
    ax.set_aspect('equal')
    #ax.set_xlabel('Neuron grid x-axis')
    #ax.set_ylabel('Neuron grid y-axis')
    #plt.show()

    return

def draw_landscape_quiver_rev(d, ax):
    [X, Y, u, v] = d

    #fig, ax = plt.subplots()
    q = ax.quiver(X, Y, v, u, pivot='mid')
    ax.set_aspect('equal')
    #ax.set_xlabel('Neuron grid x-axis')
    #ax.set_ylabel('Neuron grid y-axis')
    #plt.show()

    return

def add_colorbar(im, ax):
    ax1_divider = make_axes_locatable(ax)
    # add an axes to the right of the main axes.
    cax = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = plt.colorbar(im, cax=cax)
    return cbar, cax

def make_animation(ts, gids, nrow, ncol, nN, t_res, qdata):
    simtime = ts[-1]

    ts_bins = np.arange(0., simtime + 1, t_res)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    fig, ax = pl.subplots(1)
    a = anim.images(ax, hh, qdata, vmin=0, vmax=np.max(hh))
    return a

def make_animation_old(ts, gids, nrow, ncol, nN, t_res):
    simtime = ts[-1]

    ts_bins = np.arange(0., simtime + 1, t_res)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    fig, ax = pl.subplots(1)
    a = anim.images_old(ax, hh, 0,0,vmin=0, vmax=np.max(hh))
    return a

def make_animation_threshold_unif(ts, gids, nrow, ncol, nN, t_res, threshold):
    simtime = ts[-1]

    ts_bins = np.arange(0., simtime + 1, t_res)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    hh = np.where(hh > threshold, hh, 0)
    hh = np.where(hh == 0, hh, 1.)

    fig, ax = pl.subplots(1)
    a = anim.images_old(ax, hh, vmin=0, vmax=np.max(hh))
    return a

def make_animation_old_label(ts, gids, nrow, ncol, nN, t_res, text_all):
    simtime = ts[-1]

    ts_bins = np.arange(0., simtime + 1, t_res)
    h = np.histogram2d(ts, gids, bins=[ts_bins, range(nN + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    text = []
    for t in text_all:
        l = (t[2] - t[1])/t_res
        text += [t[0]] * int(l)

    print(len(ts_bins), len(text))

    fig, ax = pl.subplots(1)
    a = anim.images_old_label(ax, hh, t_res, text, vmin=0, vmax=np.max(hh))
    return a


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

def read_AM_normal_sparse(save_AM_parameters):
    [AM_address, fname] = save_AM_parameters

    AM = sparse.load_npz(AM_address + fname + 'normalAM.npz').toarray()
    landscape = sparse.load_npz(AM_address + fname + 'landscape.npz').toarray()
    nrowL = np.sqrt(int(np.size(landscape)))
     
    return [AM, landscape, int(nrowL)]

def get_coor(n):
    x = []
    y = []
    for i in n:
        ix = i // 120
        iy = i % 120
        x.append(ix)
        y.append(iy)
    return x,y
