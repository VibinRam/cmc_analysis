import matplotlib.pyplot as plt
from cmc_analysis.utils import *
import scipy.optimize as scp
from matplotlib.colors import LogNorm
import os

LEGEND_KWARGS = dict(
    fontsize=6,
    framealpha=0.25,
    borderpad=0.2,
    labelspacing=0.2,
    handletextpad=0.3,
    borderaxespad=0.2,
)

MERGER_MARKER_SIZE = 3
FORMATION_MARKER_SIZE = 6
RIGHT_HIST_BINS = 10

def plot_bh_worldlines(
        wids,
        bh_worldlines,
        bh_id_to_wid,
        linestyle='--',
        write_wid=False,
        add_all_related=False,
        save_fig=True,
        save_file="test_plot.pdf",
        **kwargs,
        ):
    
    fig, ax = plt.subplots()

    wids = set(wids)

    if add_all_related:
        
        all_wids = set()

        for wid in wids:

            all_wids.update(
                get_all_related_wid(
                    wid, bh_worldlines, bh_id_to_wid
                )
            )

        wids = all_wids

    for wid in wids:

        bh_worldlines[wid].add_time_mass_graph(ax, linestyle=linestyle, write_wid=write_wid)

    ax.scatter(
        [],[], edgecolors=EVENT_TO_COLOR["escape"],
        facecolors='none', label="escape"
        )
    
    EVENT_TO_COLOR_NOESC = {
        key : value
        for key, value in EVENT_TO_COLOR.items()
        if key != "escape"
    }

    for event_type in EVENT_TO_COLOR_NOESC.keys():
        ax.plot(
                [],[], color=EVENT_TO_COLOR_NOESC[event_type],
                marker='*', label=event_type,
                linestyle=None
            )
        
    ax.set_xlabel('time (code unit)')
    ax.set_ylabel(r'mass ($M_\odot$)')
    ax.legend(**LEGEND_KWARGS)
    ax.set(**kwargs)
        
    fig.savefig(save_file)

    print(f"Plotted world lines {wids} to {save_file}")

    return ax
        
def plot_bh_mergers(all_mergers, save_path = ""):

    fig = plt.figure(figsize=(8,6))

    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0)

    ax = fig.add_subplot(gs[0])        # main plot
    ax_right = fig.add_subplot(gs[1], sharey=ax)  # histogram

    fig2 = plt.figure(figsize=(8,6))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0)
    ax2 = fig2.add_subplot(gs2[0])
    ax2_right = fig2.add_subplot(gs2[1])

    fig3 = plt.figure(figsize=(8,6))
    gs3 = fig3.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0)
    ax3 = fig3.add_subplot(gs3[0, 0])
    ax3_right = fig3.add_subplot(gs3[0, 1], sharey=ax3)
    ax3_bottom = ax3.inset_axes([0.0, 0.0, 1.0, 0.22])
    fig4, ax4 = plt.subplots()

    mass_ratio_list = []
    hg_mergers = []
    g1_C_mergers = []
    g1_A_mergers = []
    g1_mergers = []
    hg_spins = []
    g1_C_spins = []
    g1_A_spins = []
    g1_spins = []

    for wid, merger_info in all_mergers.items():

        for nth in range(len(merger_info['times'])):
            
            time = merger_info['times'][nth]

            host_mass = merger_info['host_masses'][nth]

            partner_mass = merger_info['partner_masses'][nth]

            merger_type = merger_info['merger_types'][nth]

            host_spin = merger_info['host_spins'][nth]

            partner_spin = merger_info['partner_spins'][nth]

            cos_tilt_1, cos_tilt_2 = get_isotropic_tilts()

            effective_spin = calc_effective_spin(
                        host_mass, partner_mass, 
                        host_spin, partner_spin,
                        cos_tilt_1, cos_tilt_2) 

            chirp_mass = calc_chirp_mass(host_mass, partner_mass)

            semi_maj = merger_info['semi_majs'][nth]
            eccentricity = merger_info['eccentricitys'][nth]

            marker = 'o'

            if 'HG' in merger_type:
                color = 'red'
                hg_mergers.append([time, chirp_mass])
                hg_spins.append(effective_spin)
            elif ('C' in merger_type):
                color = 'blue'
                g1_C_mergers.append([time, chirp_mass])
                g1_C_spins.append(effective_spin)
            elif ('A' in merger_type):
                color = 'orange'
                g1_A_mergers.append([time, chirp_mass])
                g1_A_spins.append(effective_spin)
            else:
                color = 'green'
                g1_mergers.append([time, chirp_mass])   
                g1_spins.append(effective_spin)
            
            if host_mass > partner_mass:
                mass_ratio = partner_mass/host_mass
            else:
                mass_ratio = host_mass/partner_mass

            mass_ratio_list.append(mass_ratio)

            ax.scatter(time, chirp_mass, marker=marker, color=color, s=MERGER_MARKER_SIZE)
            ax2.scatter(chirp_mass, mass_ratio, marker=marker, color=color, s=MERGER_MARKER_SIZE)
            ax3.scatter(chirp_mass, effective_spin, marker=marker, color=color, s=MERGER_MARKER_SIZE)
            ax4.scatter(chirp_mass, eccentricity, marker=marker, color=color, s=MERGER_MARKER_SIZE)

            if 'E' in merger_type:
                ax.scatter(time, chirp_mass, facecolors='none',
                           edgecolors='pink')
                ax2.scatter(chirp_mass, mass_ratio, facecolors='none',
                           edgecolors='pink')
                ax3.scatter(chirp_mass, effective_spin, facecolors='none',
                           edgecolors='pink')
                ax4.scatter(chirp_mass, eccentricity, facecolors='none',
                           edgecolors='pink')

    ax.scatter([], [], color='red', label="Higher Gen")
    ax.scatter([], [], color='blue', label="1G + BH-STAR Collition")
    ax.scatter([], [], color='orange', label="1G + Wind fed Accretion")
    ax.scatter([], [], color='green', label="1G")
    ax.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')
    ax.set_xlabel('time (code unit)')
    ax.set_ylabel(r'chirp mass ($M_\odot$)')

    if hg_mergers:
        ax.hist(list(zip(*hg_mergers))[0], bins=20, color='red', histtype='step')
    if g1_C_mergers:
        ax.hist(list(zip(*g1_C_mergers))[0], bins=20, color='blue', histtype='step')
    if g1_A_mergers:
        ax.hist(list(zip(*g1_A_mergers))[0], bins=20, color='orange', histtype='step')
    if g1_mergers:
        ax.hist(list(zip(*g1_mergers))[0], bins=20, color='green', histtype='step')

    if hg_mergers:
        ax_right.hist(list(zip(*hg_mergers))[1], bins=RIGHT_HIST_BINS, color='red', histtype='step', orientation='horizontal')
    if g1_C_mergers:
        ax_right.hist(list(zip(*g1_C_mergers))[1], bins=RIGHT_HIST_BINS, color='blue', histtype='step', orientation='horizontal')
    if g1_A_mergers:
        ax_right.hist(list(zip(*g1_A_mergers))[1], bins=RIGHT_HIST_BINS, color='orange', histtype='step', orientation='horizontal')
    if g1_mergers:
        ax_right.hist(list(zip(*g1_mergers))[1], bins=RIGHT_HIST_BINS, color='green', histtype='step', orientation='horizontal')

    ax_right.spines['left'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.xaxis.set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax.legend(**LEGEND_KWARGS)

    ax2.scatter([], [], color='red', label="Higher Gen")
    ax2.scatter([], [], color='blue', label="1G + BH-STAR Collition")
    ax2.scatter([], [], color='orange', label="1G + Wind fed Accretion")
    ax2.scatter([], [], color='green', label="1G")
    ax2.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')

    ax2_right.hist(mass_ratio_list, bins=RIGHT_HIST_BINS, histtype='step', orientation='horizontal')

    ax2_right.spines['left'].set_visible(False)
    ax2_right.spines['top'].set_visible(False)
    ax2_right.spines['bottom'].set_visible(False)
    ax2_right.spines['right'].set_visible(False)
    ax2_right.xaxis.set_visible(False)
    ax2_right.yaxis.set_visible(False)
    
    ax2.set_xlabel(r'chirp mass ($M_\odot$)')
    ax2.set_ylabel('mass ratio')
    ax2.legend(**LEGEND_KWARGS)

    ax3.scatter([], [], color='red', label="Higher Gen")
    ax3.scatter([], [], color='blue', label="1G + BH-STAR Collition")
    ax3.scatter([], [], color='orange', label="1G + Wind fed Accretion")
    ax3.scatter([], [], color='green', label="1G")
    ax3.scatter([], [], facecolors='none', edgecolors='pink',
               label='Escape merger')    
    ax3.tick_params(axis='x', bottom=False, labelbottom=False)

    ax3_bottom.set_facecolor('none')
    ax3_bottom.patch.set_alpha(0.0)
    ax3_bottom.set_zorder(5)

    if hg_mergers:
        ax3_bottom.hist(list(zip(*hg_mergers))[1], bins=20, color='red', histtype='step')
    if g1_C_mergers:
        ax3_bottom.hist(list(zip(*g1_C_mergers))[1], bins=20, color='blue', histtype='step')
    if g1_A_mergers:
        ax3_bottom.hist(list(zip(*g1_A_mergers))[1], bins=20, color='orange', histtype='step')
    if g1_mergers:
        ax3_bottom.hist(list(zip(*g1_mergers))[1], bins=20, color='green', histtype='step')

    ax3_bottom.spines['left'].set_visible(False)
    ax3_bottom.spines['top'].set_visible(False)
    ax3_bottom.spines['right'].set_visible(False)
    ax3_bottom.yaxis.set_visible(False)
    ax3_bottom.spines['bottom'].set_visible(True)
    ax3_bottom.tick_params(axis='x', bottom=True, labelbottom=True)

    if hg_spins:
        ax3_right.hist(hg_spins, bins=RIGHT_HIST_BINS, color='red', histtype='step', orientation='horizontal')
    if g1_C_spins:
        ax3_right.hist(g1_C_spins, bins=RIGHT_HIST_BINS, color='blue', histtype='step', orientation='horizontal')
    if g1_A_spins:
        ax3_right.hist(g1_A_spins, bins=RIGHT_HIST_BINS, color='orange', histtype='step', orientation='horizontal')
    if g1_spins:
        ax3_right.hist(g1_spins, bins=RIGHT_HIST_BINS, color='green', histtype='step', orientation='horizontal')

    ax3_right.spines['left'].set_visible(False)
    ax3_right.spines['top'].set_visible(False)
    ax3_right.spines['bottom'].set_visible(False)
    ax3_right.spines['right'].set_visible(False)
    ax3_right.xaxis.set_visible(False)
    ax3_right.yaxis.set_visible(False)

    ax3.set_ylabel("Effective spin")
    ax3.legend(**LEGEND_KWARGS)

    ax3_bottom.set_xlabel('chirp mass', labelpad=6)

    ax4.scatter([], [], color='red', label="Higher Gen")
    ax4.scatter([], [], color='blue', label="1G + BH-STAR Collition")
    ax4.scatter([], [], color='orange', label="1G + Wind fed Accretion")
    ax4.scatter([], [], color='green', label="1G")
    ax4.scatter([], [], facecolors='none', edgecolors='pink',
               label='Escape merger')
    ax4.set_xlabel('chirp mass')
    ax4.set_ylabel('eccentricity')  
    ax4.legend(**LEGEND_KWARGS)

    fig.savefig(os.path.join(save_path, 'merger_plot.pdf'))
    fig2.savefig(os.path.join(save_path, 'merger_plot2.pdf'))
    fig3.savefig(os.path.join(save_path, 'merger_plot3.pdf'))
    fig4.savefig(os.path.join(save_path, 'merger_plot4.pdf'))

    # ax3.set_ylim(-1e-4, 1e-4)
    ax3.set_yscale('log')
    fig3.savefig(os.path.join(save_path, 'merger_plot3_lowspin.pdf'))

    #Printing count report
    print(f"hg mergers - {len(hg_spins)}")
    print(f"1G + BH-STAR Collision - {len(g1_C_spins)}")
    print(f"1G + Wind fed Accretion - {len(g1_A_spins)}")
    print(f"1G - {len(g1_spins)}")
    print(f"Total - {len(hg_spins) + len(g1_C_spins) + len(g1_A_spins) + len(g1_spins)}")

    return ax

def plot_bh_formations(bh_formations):

    fig, ax = plt.subplots(figsize=(6, 6))
    labeled_points = False

    for bh_id in bh_formations.keys():
        birth_time = bh_formations[bh_id]['time']
        m_progenitor = bh_formations[bh_id]['m_progenitor']
        m_bh = bh_formations[bh_id]['m_bh']

        ax.plot(
            [birth_time, birth_time],
            [m_progenitor, m_bh],
            color='k',
            linewidth=0.5,
            linestyle='dashed'
        )
        ax.scatter(
            birth_time,
            m_progenitor,
            color='tab:orange',
            s=FORMATION_MARKER_SIZE,
            label='Progenitor mass' if not labeled_points else None,
        )
        ax.scatter(
            birth_time,
            m_bh,
            color='tab:blue',
            s=FORMATION_MARKER_SIZE,
            label='BH mass' if not labeled_points else None,
        )
        labeled_points = True

    ax.set_xlabel('time')
    ax.set_ylabel('mass')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3, 1e-1])
    if labeled_points:
        ax.legend(**LEGEND_KWARGS)

    fig.savefig('formations_plot1.pdf')
    return ax
        
def plot_chirp_spin_2D(all_mergers):
    
    mass_ratio_list = []
    hg_mergers = []
    g1_C_mergers = []
    g1_A_mergers = []
    g1_mergers = []
    hg_spins = []
    g1_C_spins = []
    g1_A_spins = []
    g1_spins = []

    for i in range(100):

        for wid, merger_info in all_mergers.items():

            for nth in range(len(merger_info['times'])):
                
                time = merger_info['times'][nth]

                host_mass = merger_info['host_masses'][nth]

                partner_mass = merger_info['partner_masses'][nth]

                merger_type = merger_info['merger_types'][nth]

                host_spin = merger_info['host_spins'][nth]

                partner_spin = merger_info['partner_spins'][nth]

                cos_tilt_1, cos_tilt_2 = get_isotropic_tilts()

                effective_spin = calc_effective_spin(
                            host_mass, partner_mass, 
                            host_spin, partner_spin,
                            cos_tilt_1, cos_tilt_2) 

                chirp_mass = calc_chirp_mass(host_mass, partner_mass)

                semi_maj = merger_info['semi_majs'][nth]
                eccentricity = merger_info['eccentricitys'][nth]

                marker = 'o'

                if 'HG' in merger_type:
                    color = 'red'
                    hg_mergers.append([time, chirp_mass])
                    hg_spins.append(effective_spin)
                elif ('C' in merger_type):
                    color = 'blue'
                    g1_C_mergers.append([time, chirp_mass])
                    g1_C_spins.append(effective_spin)
                elif ('A' in merger_type):
                    color = 'orange'
                    g1_A_mergers.append([time, chirp_mass])
                    g1_A_spins.append(effective_spin)
                else:
                    color = 'green'
                    g1_mergers.append([time, chirp_mass])   
                    g1_spins.append(effective_spin)
                
                if host_mass > partner_mass:
                    mass_ratio = partner_mass/host_mass
                else:
                    mass_ratio = host_mass/partner_mass

                mass_ratio_list.append(mass_ratio)

    master_chirp_mass_list = np.concatenate(
        (
            np.array(hg_mergers)[:,0],
            np.array(g1_C_mergers)[:,0],
            np.array(g1_A_mergers)[:,0],
            np.array(g1_mergers)[:,0]
        ),
        axis=0
    )
    
    master_spin_list = np.concatenate(
        (
            np.array(hg_spins),
            np.array(g1_C_spins),
            np.array(g1_A_spins),
            np.array(g1_spins)
        ),
        axis=0
    )


    fig, ax = plt.subplots()
    _,_,_,im = ax.hist2d(
        master_chirp_mass_list, 
        master_spin_list, 
        bins=100,
        norm=LogNorm()
        )
    fig.colorbar(im, ax=ax)

    fig.savefig("chirp_spin_2D.pdf")
    plt.close(fig)