import matplotlib.pyplot as plt
from cmc_analysis.utils import *
import scipy.optimize as scp

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
    ax.legend(fontsize=8)
    ax.set(**kwargs)
        
    fig.savefig(save_file)

    print(f"Plotted world lines {wids} to {save_file}")

    return ax
        
def plot_bh_mergers(all_mergers):

    fig = plt.figure(figsize=(8,6))

    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0)

    ax = fig.add_subplot(gs[0])        # main plot
    ax_right = fig.add_subplot(gs[1], sharey=ax)  # histogram

    fig2 = plt.figure(figsize=(8,6))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0)
    ax2 = fig2.add_subplot(gs2[0])
    ax2_right = fig2.add_subplot(gs2[1])

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    mass_ratio_list = []
    hg_mergers = []
    g1_spec_mergers = []
    g1_mergers = []

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

            if 'HG' in merger_type:
                color = 'red'
                hg_mergers.append([time, chirp_mass])
            elif ('A' in merger_type) or ('C' in merger_type):
                color = 'blue'
                g1_spec_mergers.append([time, chirp_mass])
            else:
                color = 'green'
                g1_mergers.append([time, chirp_mass])
            
            if host_mass > partner_mass:
                mass_ratio = partner_mass/host_mass
            else:
                mass_ratio = host_mass/partner_mass

            mass_ratio_list.append(mass_ratio)

            ax.scatter(time, chirp_mass, color=color, s=5)
            ax2.scatter(chirp_mass, mass_ratio, color=color, s=5)
            ax3.scatter(chirp_mass, effective_spin, color=color, s=5)
            ax4.scatter(chirp_mass, eccentricity, color=color, s=5)

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
    ax.scatter([], [], color='blue', label="1G special")
    ax.scatter([], [], color='green', label="1G")
    ax.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')
    ax.set_xlabel('time (code unit)')
    ax.set_ylabel(r'chirp mass ($M_\odot$)')

    ax.hist(list(zip(*hg_mergers))[0], bins=20, color='red', histtype='step')
    ax.hist(list(zip(*g1_spec_mergers))[0], bins=20, color='blue', histtype='step')
    ax.hist(list(zip(*g1_mergers))[0], bins=20, color='green', histtype='step')

    ax_right.hist(list(zip(*hg_mergers))[1], bins=20, color='red', histtype='step', orientation='horizontal')
    ax_right.hist(list(zip(*g1_spec_mergers))[1], bins=20, color='blue', histtype='step', orientation='horizontal')
    ax_right.hist(list(zip(*g1_mergers))[1], bins=20, color='green', histtype='step', orientation='horizontal')

    ax_right.spines['left'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.xaxis.set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax.legend(fontsize=8)

    ax2.scatter([], [], color='red', label="Higher Gen")
    ax2.scatter([], [], color='blue', label="1G special")
    ax2.scatter([], [], color='green', label="1G")
    ax2.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')

    ax2_right.hist(mass_ratio_list, bins=20, histtype='step', orientation='horizontal')

    ax2_right.spines['left'].set_visible(False)
    ax2_right.spines['top'].set_visible(False)
    ax2_right.spines['bottom'].set_visible(False)
    ax2_right.spines['right'].set_visible(False)
    ax2_right.xaxis.set_visible(False)
    ax2_right.yaxis.set_visible(False)
    
    ax2.set_xlabel(r'chirp mass ($M_\odot$)')
    ax2.set_ylabel('mass ratio')
    ax2.legend(fontsize=8)

    ax3.scatter([], [], color='red', label="Higher Gen")
    ax3.scatter([], [], color='blue', label="1G special")
    ax3.scatter([], [], color='green', label="1G")
    ax3.scatter([], [], facecolors='none', edgecolors='pink',
               label='Escape merger')    

    ax3.set_xlabel('chirp mass')
    ax3.set_ylabel("Effective spin")
    ax3.legend()

    ax4.scatter([], [], color='red', label="Higher Gen")
    ax4.scatter([], [], color='blue', label="1G special")
    ax4.scatter([], [], color='green', label="1G")
    ax4.scatter([], [], facecolors='none', edgecolors='pink',
               label='Escape merger')
    ax4.set_xlabel('chirp mass')
    ax4.set_ylabel('eccentricity')  
    ax4.legend()  

    fig.savefig('merger_plot.pdf')
    fig2.savefig('merger_plot2.pdf')
    fig3.savefig('merger_plot3.pdf')
    fig4.savefig('merger_plot4.pdf')

    return ax
