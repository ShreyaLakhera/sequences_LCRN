import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import new_plot_scripts.plotting_functions as pf
import new_sim_scripts.simulation_functions as sf
import time
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import pdist
import scipy.stats as stats
from scipy import signal

def correlation_distance(t1, t2):
    d = 1 - np.corrcoef(t1,t2)[0,1]
    return d

def showClustersGrid(clusters, N_selected):
    nrow = ncol = 120
    A = np.zeros((nrow*ncol))
    for ik,ic in enumerate(clusters):
        for iN in ic:
            N_id = N_selected[iN]
            A[N_id] = (ik + 1)*10
    A = np.reshape(A, (nrow, ncol))
    return A

def runMetaKMeans(data, metric, k, n_trials):
    print("\nRunning k-means {} times...".format(n_trials))
    all_clusters = []
    for nt in range(n_trials):
        clusters = runKMeans(data, k, metric)
        all_clusters.append(clusters)        

    nN_sel = np.shape(data)[0]
    print("Counting co-occurrences...")
    counts_mat = np.zeros((nN_sel, nN_sel))
    for i in range(nN_sel):
        counts_mat[i,i] = n_trials
        for j in range(i+1,nN_sel):
            cij = 0
            for t in range(n_trials):
                cluster = all_clusters[t]
                ci = -1
                cj = -1
                for _ic in range(k):
                    if i in cluster[_ic]:
                        ci = _ic
                    if j in cluster[_ic]:
                        cj = _ic
                    if (ci != -1) and (cj != -1):
                        break
                if ci == cj:
                    cij += 1
            counts_mat[i,j] = cij
            counts_mat[j,i] = cij

    return counts_mat

def runKMeans(data, k, user_metric):
    initial_centers = kmeans_plusplus_initializer(data,k).initialize()
    kmeans_instance = kmeans(data, initial_centers, metric=user_metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    return clusters

def runKMedoid(distance_matrix, k):
    nN_nz = np.shape(distance_matrix)[0]
    initial_medoids = np.random.choice(np.arange(nN_nz), k).tolist()
    kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    return clusters


def DunnsIndexMatrix(clusters, distance_matrix):

    k = len(clusters)

    min_inter_d = []
    max_intra_d = []
    for i in range(k):
        c = clusters[i]
        dist = distance_matrix[c,:][:,c]
        max_intra_d.append(np.max(dist))
        for j in range(k):
            if j != i:
                cj = clusters[j]
                dist = distance_matrix[c,:][:,cj]
                min_inter_d.append(np.min(dist))

    dmin = min(min_inter_d)
    dmax = max(max_intra_d)

    dunns_index = dmin/dmax

    return dunns_index


def optimiseKMedoid(distance_matrix, k, n_trials, di_matrix):

    all_clusters = []
    all_DI = []

    print("\nRunning K-Medoid for {} runs to optimise...".format(n_trials))

    for t in range(n_trials):
        clusters = runKMedoid(distance_matrix, k)
        DI = DunnsIndexMatrix(clusters, di_matrix)
        all_clusters.append(clusters)
        all_DI.append(DI)

    i = np.argmax(np.array(all_DI))
    best_clustering = all_clusters[i]
    
    print("Best clustering found.")

    return best_clustering, all_DI[i]

def optimiseKMean(data, user_metric, k, n_trials, di_matrix):

    all_clusters = []
    all_DI = []

    print("\nRunning K-Means for {} runs to optimise...".format(n_trials))

    for t in range(n_trials):
        clusters = runKMeans(data, k, user_metric)
        DI = DunnsIndexMatrix(clusters, di_matrix)
        all_clusters.append(clusters)
        all_DI.append(DI)

    i = np.argmax(np.array(all_DI))
    best_clustering = all_clusters[i]
    
    print("Best clustering found.")

    return best_clustering, all_DI[i]

def find_working_clusters(counts_matrix, threshold, ntrials, di_matrix):
    N_all = np.arange(np.shape(counts_matrix)[0]).astype(int)
    clusters_all = []
    DI_all = []
    for i_t in range(ntrials):
        ungrouped_neurons = list(N_all)
        working_clusters = []

        while len(ungrouped_neurons) > 0:
            
            temp_group = [np.random.choice(ungrouped_neurons).astype(int)]
            ungrouped_neurons.remove(temp_group[0])

            for i,n in enumerate(ungrouped_neurons):
                new_group = temp_group + [n]
                group_counts_matrix = counts_matrix[new_group,:][:,new_group]
                
                #if len(np.where(group_counts_matrix < threshold)[0]) == 0:
                if len(np.where(group_counts_matrix < threshold)[0]) == 0:
                    temp_group = new_group
                
            for n in temp_group[1:]:
                ungrouped_neurons.remove(n)
                    
            working_clusters.append(temp_group)
        
        if len(working_clusters) > 1:        
            di = DunnsIndexMatrix(working_clusters, di_matrix)
            DI_all.append(di)
            clusters_all.append(working_clusters)

    if len(clusters_all) > 0:
        max_DI_id = np.argmax(np.array(DI_all))
        final_cluster = clusters_all[max_DI_id]
    else:
        max_DI_id = -1
        final_cluster = []

    return final_cluster, max_DI_id

def merge_working_clusters(working_clusters, di_matrix, FR_corr):

    # find most correlated pair c1 and c2 
    # merge c1 and c2 -> new clustering
    # calculate DI of new clustering
    # if new DI > original DI => accept clustering
    # original clustering = new clustering
    # original DI = new DI
    # calculate correlation matrix between clusters

    # find DI of original clustering
    # Calculate correlation matrix between clusters
    original_DI = DunnsIndexMatrix(working_clusters, di_matrix)
    original_cluster_correlations = calculate_corr_matrix_clusters(working_clusters, FR_corr)
    print(np.shape(original_cluster_correlations))
    tr = True

    while tr:
        
        OC = np.triu(original_cluster_correlations, k=1)
        ind = np.unravel_index(np.argsort(OC, axis=None)[::-1], OC.shape)
        l = int(np.size(OC)/2)
        
        for i in range(l):
            #[_c1, _c2] = np.unravel_index(np.argmax(OC, axis=None), OC.shape)
            _c1, _c2 = ind[0][i], ind[1][i]
           # print(_c1, _c2)
            
            if i == l-1:
                tr = False

            if _c1 == _c2:
                continue
            
            new_cluster = [working_clusters[_c1] + working_clusters[_c2]]
            old_cluster = working_clusters.copy()
            old_cluster.remove(working_clusters[_c1])
            old_cluster.remove(working_clusters[_c2])
            new_clustering = new_cluster + old_cluster
            
            if len(new_clustering) == 1:
                continue

            new_DI = DunnsIndexMatrix(new_clustering, di_matrix)
            
            if new_DI > original_DI:
                working_clusters = new_clustering
                original_DI = new_DI
                original_cluster_correlations = calculate_corr_matrix_clusters(working_clusters, FR_corr)
                #print('Merged')
                tr = True
                break

    return working_clusters

def calculate_corr_matrix_clusters(working_clusters, init_corr_matrix):

    n = len(working_clusters)
    corr_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n-1):
            wc1, wc2 = working_clusters[i], working_clusters[j]
            all_corr = init_corr_matrix[wc1,:][:,wc2]       # should I use distance matrix based on counts instead?
            corr_val = np.mean(all_corr)    # decide what
            corr_matrix[i, j] = corr_val
            corr_matrix[j, i] = corr_val

    return corr_matrix

def apply_gaussian_filter(FR_time, g_length, g_sigma):
    w = signal.gaussian(g_length, g_sigma)
    nN = 60
    new_FR = []
    for x in range(nN):
        nx = np.convolve(FR_time[x,:], w/w.sum(), 'valid')
        new_FR.append(nx.tolist())
    return np.array(new_FR)

def apply_movingaverage_filter(FR_time, length):
    w = np.ones(length)
    new_FR = np.convolve(FR_time, w/w.sum(), 'valid')
    return new_FR

def apply_exponential_filter(FR_time, length, tau):
    w = signal.exponential(length, tau=tau)         
    # check the asymmetric one
    #tau2 = -(length - 1)/np.log(0.01)
    #w = signal.exponential(length, 0, tau2, sym=False)
    nN = 60
    new_FR = []
    for x in range(nN):
        nx = np.convolve(FR_time[x,:], w/w.sum(), 'valid')
        new_FR.append(nx.tolist())
    return np.array(new_FR)

def apply_doubleexponential_filter(FR_time, length, tau):
    w = signal.exponential(length, tau=tau)
    nN = 60
    new_FR = []
    for x in range(nN):
        nx = np.convolve(FR_time[x,:], w/w.sum(), 'valid')
        new_FR.append(nx.tolist())
    return np.array(new_FR)

if __name__ == "__main__":
    
    """
    """
    # Init parameters

    nrowE = ncolE = 120
    nrowI = ncolI = 60

    nE = nrowE * ncolE
    nI = nrowI * ncolI

    offsetE = 1
    offsetI = nE + 1

    address = sys.argv[1]
    nfiles = 1

    # Load 

    ts, ids = pf.get_spiketimes(address, nfiles)

    idE = ids < nE + offsetE
    tsE, idsE = ts[idE], ids[idE]
    tsI, idsI = ts[~idE], ids[~idE]

    t_start = 0.
    t_end = 40000.
   
    tsE1, idsE1 = pf.getSpikesTimeInterval(tsE, idsE, t_start, t_end)
    FR_entiretime = pf.calculate_firing_rate(tsE1, idsE1, nE, t_end)
    
    nN_nz = 60         # Number of sample neurons

    fr_threshold = float(sys.argv[2]) * np.max(FR_entiretime)

    print(np.shape(FR_entiretime))
    #print(np.unique(FR_entiretime))
    #plt.hist(FR_entiretime, bins=100)
    #plt.axvline(fr_threshold, 0, 2,c='k')
    #plt.show()
    
    t_step = 50.
    N_nonzero_all = np.where(FR_entiretime > fr_threshold)[0]

    k = 5
    
    #xfil = 'gaussian'
    #g_len = 10
    #g_sig = 3
    
    xfil = 'exponential'
    length = 10
    tau = 1
    
    bad_sample = True
    while bad_sample:

        print('Selecting a sample of {} neurons from {} neurons'.format(nN_nz, int(np.shape(N_nonzero_all)[0])))
        N_nonzero = np.random.choice(N_nonzero_all, nN_nz, replace=False)
        ts_nz, ids_nz = pf.getSpiketimesGivenNeurons(tsE1, idsE1, N_nonzero)
        ts_nz, ids_nz = pf.reassignNeuronID(ts_nz, ids_nz, N_nonzero, np.arange(nN_nz))
        
        FR_nz = pf.get_FR_time(ts_nz, ids_nz, nN_nz, t_step, t_start, t_end)
        FR_nz = apply_exponential_filter(FR_nz, length, tau) #change
        #FR_nz = apply_gaussian_filter(FR_nz, g_len, g_sig) #change
        #FR_nz = apply_movingaverage_filter(FR_nz, 10) #change
        FR_corr = np.corrcoef(FR_nz)
        print(np.unique(FR_corr)[-5:])
        nan_val = np.count_nonzero(np.isnan(FR_corr))
        if nan_val == 0:
            bad_sample = False
            print("Found good sample\n")

    for t_step in [10., 25., 50., 100.]:
        for length in [5,7,9,11]:
            for tau in [1,3,5]:

                xfilter = '{}length{}tau{}'.format(xfil, length, tau)
                print('{} {} {}'.format(xfil, length, tau))
                
                #xfilter = '{}gl{}gs{}'.format(xfil, g_len, g_sig)
                #print('{} {} {}'.format(xfil, g_len, g_sig))
            
                FR_nz1 = pf.get_FR_time(ts_nz, ids_nz, nN_nz, t_step, t_start, t_end)
                FR_nz = apply_exponential_filter(FR_nz1, length, tau) #change
                #FR_nz = apply_gaussian_filter(FR_nz1, g_len, g_sig)
                #FR_nz = apply_movingaverage_filter(FR_nz, 10) #change
                FR_corr = np.corrcoef(FR_nz)

                # clustering

                FR = FR_nz
                n_trials = 20
                n_trials_wc = 300
                
                corr_metric = distance_metric(type_metric.USER_DEFINED, func=correlation_distance)
                distance_corr = 1 - FR_corr

                cmap = mpl.colors.ListedColormap(['white', 'yellow', 'blue', 'red', 'green', 'black'])
                bounds = [-10, 0.01, 10, 20, 30, 40, 50]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                
                #ufrc, ucfrc = np.unique(np.reshape(FR_corr, nN_nz**2,1), return_counts=True)

                counts_matrix = runMetaKMeans(FR_nz, corr_metric, k, n_trials)
                
                di_matrix_counts = n_trials - counts_matrix#_threshold
                di_matrix = 1 - FR_corr
               
                if len(np.unique(counts_matrix)) == 1:
                    print(np.unique(counts_matrix))
                    continue

                print("Calculating working clusters")
                threshold = float(sys.argv[3]) * np.mean(counts_matrix)
                #threshold = np.mean(counts_matrix) + float(sys.argv[3])*np.std(counts_matrix)
                print("Mean of counts matrix = {}, threshold = {}".format(round(np.mean(counts_matrix), 5), round(threshold,5)))
                working_clusters, max_di = find_working_clusters(counts_matrix, threshold, n_trials_wc, di_matrix_counts)
                print('Number of working clusters = {}\nMaximum DI = {}\n'.format(len(working_clusters), max_di))
                
                print('Merging clusters')
                merged_clusters = merge_working_clusters(working_clusters, di_matrix_counts, FR_corr)
                print('Number of clusters after merging = {}\n'.format(len(merged_clusters)))

                cl_order = []
                for i in range(len(merged_clusters)):
                    cl_order += merged_clusters[i]

                new_counts_matrix = counts_matrix[cl_order,:][:,cl_order]
                clustered_corr = FR_corr[cl_order,:][:,cl_order]

                # Show spatial positions of clusters
                plt.figure(figsize=(6,6))
                colors = ['xkcd:sky blue', 'xkcd:orange', 'xkcd:green', 'xkcd:magenta', 'xkcd:yellow']
                mark = ['o','s','^']
                clusters = merged_clusters
                for i,c in enumerate(clusters):
                    n_id = [N_nonzero[j] for j in c]
                    x,y = pf.get_coor(n_id)
                    plt.scatter(y,x, label='Cluster {}'.format(i))
                #plt.imshow(A, cmap=cmap, norm=norm)
                plt.legend(loc='upper right')
                plt.xlabel('Neuron grid x-axis')
                plt.ylabel('Neuron grid y-axis')
                plt.xlim([0,120])
                plt.ylim([0,120])
                plt.title("Spatial positions of clusters")
                plt.tight_layout()
                plt.savefig('{}clusters_spatial_positions_k{}_dt{}_{}'.format(address, k, int(t_step), xfilter))
                #plt.show()
                plt.close()

                # Show distribution of correlations in inter- vs intra- clusters
                plt.figure()
                FR_corr_remaining = FR_corr.copy()
                for i,c in enumerate(clusters):
                    corrs = FR_corr[c,:][:,c]
                    freq, be = np.histogram(corrs)
                    bw = (be[1] - be[0])/2
                    bc = be[1:] - bw
                    plt.plot(bc, freq/freq.sum(), '-', label='Inter cluster {}'.format(i))
                    FR_corr_remaining = np.setdiff1d(FR_corr_remaining, corrs) 
                    if i == len(clusters) - 1:
                        freq, be = np.histogram(FR_corr_remaining)
                        bw = (be[1] - be[0])/2
                        bc = be[1:] - bw
                        plt.plot(bc, freq/freq.sum(), '--', color='grey', label='Intra cluster')
                        FR_corr_remaining = np.setdiff1d(FR_corr_remaining, corrs) 
                plt.legend()
                plt.xlim([-1.,1.])
                plt.xlabel('Correlation Coefficient')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig('{}correlation_distributions_k{}_dt{}_{}'.format(address,k, int(t_step), xfilter))
                plt.close()

                # Clustered counts matrix
                plt.figure()
                plt.imshow(new_counts_matrix)
                plt.xlabel('Neuron ID')
                plt.ylabel('Neuron ID')
                plt.colorbar(label='Co-occurrence counts')
                plt.tight_layout()
                plt.savefig('{}clustered_counts_matrix_k{}_dt{}_{}.png'.format(address, k, int(t_step), xfilter))
                #plt.show()
                plt.close()

                # CLustered correlation matrix
                plt.figure()
                plt.imshow(clustered_corr)
                plt.xlabel('Neuron ID')
                plt.ylabel('Neuron ID')
                plt.colorbar(label='Correlation coefficient')
                plt.tight_layout()
                plt.savefig('{}clustered_corr_matrix_k{}_dt{}_{}.png'.format(address, k, int(t_step), xfilter))
                #plt.show()
                plt.close()

                # Traces example
                plt.figure()
                t_start_eg, t_end_eg = 0., 2000.
                ts_eg, ids_eg = pf.getSpikesTimeInterval(np.array(ts_nz), np.array(ids_nz), t_start_eg, t_end_eg)
                FR_eg = pf.get_FR_time(ts_eg, ids_eg, nN_nz, t_step, t_start_eg, t_end_eg)
                FR_eg1 = apply_exponential_filter(FR_eg, length, tau) #change
                [N0, N1] = np.random.randint(0, nN_nz, 2)
                FR_N1 = FR_eg[N1, :]
                FRg_N1 = FR_eg1[N1, :]
                idx_N1 = np.where(ids_eg == N1)[0]
                ts_N1 = np.array(ts_eg)[idx_N1]
                t_step_frg = (t_end_eg - t_start_eg)/len(FRg_N1)
                plt.plot(np.arange(t_start_eg, t_end_eg, t_step), FR_N1, '-', label='Firing rate')
                plt.plot(np.arange(t_start_eg, t_end_eg, t_step_frg), FRg_N1, '-', label='Firing rate gaussian filtered')
                plt.plot(ts_N1, [-1]*len(ts_N1), 'k|', label='Spiketimes')
                plt.legend()
                plt.xlabel('Time')
                plt.tight_layout()
                plt.savefig('{}example_traces_k{}_dt{}_{}.png'.format(address, k, int(t_step), xfilter))
                #plt.show()
                plt.close()

                del(working_clusters, merged_clusters, clusters)
        
        """
        for k2 in [2,3,4,5]:
            clusters, DI = optimiseKMedoid(d_matrix, k2, n_trials_op, di_matrix)
            print(DI)

            cl_order = []
            for i in range(k2):
                cl_order += clusters[i]

            new_counts_matrix = counts_matrix[cl_order,:][:,cl_order]
            print(DI)

            clustered_corr = FR_corr[cl_order,:][:,cl_order]

            #fig, ax = plt.subplots(1)#, figsize=[10,5])
            plt.figure()
            plt.imshow(clustered_corr)
            cbar = plt.colorbar()
            cbar.set_label('Correlation')
            plt.title("Correlation matrix (Clustered), k = {}".format(k2))
            plt.xlabel('Neuron index')
            plt.ylabel('Neuron index')
            plt.tight_layout()
            plt.savefig('{}clustered{}_correlation_matrix'.format(address,k2))
            #plt.show()
            plt.close()
            #ax[1].matshow(new_counts_matrix)
            #ax[1].set_title("Clustered co-occurrences, k = {}".format(k))

            # Calculate pairwise distances
            xA, yA = pf.get_coor(N_nonzero)
            all_dist = pdist(np.array([xA, yA]).T, 'euclidean')
            cluster_Ds = []
            mean_Ds = [np.mean(all_dist)]
            std_Ds = [np.std(all_dist)]
            label_Ds = ['All']

            #A = showClustersGrid(clusters, N_nonzero)
            plt.figure(figsize=(6,6))
            colors = ['xkcd:sky blue', 'xkcd:orange', 'xkcd:green', 'xkcd:magenta', 'xkcd:yellow']
            mark = ['o','s','^']
            for i,c in enumerate(clusters):
                n_id = [N_nonzero[j] for j in c]
                x,y = pf.get_coor(n_id)
                cluster_distance = pdist(np.array([x, y]).T, 'euclidean')
                cluster_Ds.append(cluster_distance)
                mean_Ds.append(np.mean(cluster_distance))
                std_Ds.append(np.std(cluster_distance))
                label_Ds.append('Cluster {}'.format(i+1))
                plt.scatter(y,x, color=colors[i], marker=mark[i%3],label='Cluster {}'.format(i))
            #plt.imshow(A, cmap=cmap, norm=norm)
            plt.legend(loc='upper right')
            plt.xlabel('Neuron grid x-axis')
            plt.ylabel('Neuron grid y-axis')
            plt.title("Spatial positions of clusters")
            plt.tight_layout()
            plt.savefig('{}clusters{}_spatial_positions'.format(address,k2))
            #plt.show()
            plt.close()

            # Show pairwise distance
            plt.bar(label_Ds, mean_Ds)
            #plt.show()
        """
