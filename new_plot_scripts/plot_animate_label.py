import sys
import numpy as np
import pylab as pl
import lib.animation as anim
import matplotlib.pyplot as plt
import new_plot_scripts.plotting_functions as pf
import new_sim_scripts.simulation_functions as sf
import glob



if __name__ == "__main__":

    nrowE = ncolE = 120
    nrowI = ncolI = 60
    nE = nrowE * ncolE
    nI = nrowI * ncolI
    nN = nE + nI

    offsetE = 1
    offsetI = nE + 1

    for asymmetry in ['EI']:
        for r in [1.0]:
            for nmbE in [250., 300., 400., 500., 550., 600.]:
                for nmbI in [nmbE]:
                    
                    print(asymmetry, r, nmbE, nmbI)

                    address = '/home/bhalla/shreyal/data/EI_brunel_{}/EI_{}_size3_std7-9-7-9_g8_J10_nmE0I0p0bE{}bI{}_base0_taumE10I10_Vth-50_ratio{}/'.format(asymmetry, asymmetry, int(nmbE), int(nmbI), r)
                    nfiles = 1
                    t_warmup = 100.
                    t_sim = 1000.
                    t_reset = 500.
                    nfiles = 1 

                    ts, gids = pf.get_spiketimes(address, nfiles)

                    ot = np.argsort(ts)
                    ts = ts[ot]
                    gids = gids[ot]

                    gidxE = gids - offsetE < nE
                    tsE, gidsE = ts[gidxE], gids[gidxE]         # Excitatory population
                    tsI, gidsI = ts[~gidxE], gids[~gidxE]       # Inhibitory population

                    tres = 10.
                   
                    print("Read spiketimes.")

                    #text_all = [
                    #        ['Warming up', 0., 100.],
                    #        ['Low uniform input', 100., 600.],
                    #        ['Reset', 600., 1100.],
                    #        ['High uniform input', 1100., 1600.],
                    #        ['Reset', 1600., 2100.],
                    #        ['Directed Input', 2100., 2600.]
                    #        ]

                    #a = pf.make_animation_old_label(tsE, gidsE - offsetE, nrowE, ncolE, nE, tres, text_all)
                    #print("Animation made.")
                    #print("Saving animation...")
                    #a.save(address+'activity_anim.mp4', fps=5, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
                    #print("Animation saved.")
                    #plt.show()

                    #offsetE = 1
                    #gidxE = ids - offsetE < nE
                    #tsE, gidsE = ts[gidxE], ids[gidxE]         # Excitatory population
                    #tsI, gidsI = ts[~gidxE], ids[~gidxE]       # Inhibitory population
                    #tres = 10.
                    tw, ts, tr = t_warmup, t_sim, t_reset
                    text_all = [
                            ['Warming up', 0., tw],
                            ['Sparse uniform', tw, tw + ts],
                    #        ['Low uniform input', tw, tw + ts],
                    #        ['Reset', tw + ts, tw + ts + tr],
                    #        ['High uniform input', tw + ts + tr, tw + 2*ts + tr],
                    #        ['Reset', tw + 2*ts + tr, tw + 2*ts + 2*tr],
                    #        ['Directed Input', tw + 2*ts + 2*tr, tw + 3*ts + 2*tr],
                    #        ['Reset', tw + 3*ts + 2*tr, tw + 3*ts + 3*tr],
                    #        ['Sparse uniform input', tw + 3*ts + 3*tr, tw + 4*ts + 3*tr],
                    #        ['Reset', tw + 4*ts + 3*tr, tw + 4*ts + 4*tr],
                    #        ['Temporal - Random', tw + 4*ts + 4*tr, tw + 5*ts + 4*tr],
                    #        ['Reset', tw + 5*ts + 4*tr, tw + 5*ts + 5*tr],
                    #        ['Temporal - Interpolated', tw + 5*ts + 5*tr, tw + 6*ts + 5*tr]
                            ]

                    #a = pf.make_animation_old_label(tsE, gidsE - offsetE, nrowE, ncolE, nE, tres, text_all)
                    a = pf.make_animation_old_label(tsI, gidsI - offsetI, nrowI, ncolI, nI, tres, text_all)
                    print("Animation made.")
                    print("Saving animation...")
                    a.save(address + 'activity_anim_I.mp4', fps=5, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
                    print("Animation saved.\n")
