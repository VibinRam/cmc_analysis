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

    print(f"Plotted world lines {wids}")
        
    fig.savefig('test_plot.pdf')

    return ax
        
def plot_bh_mergers(all_mergers):

    fig, ax = plt.subplots()

    fig2, ax2 = plt.subplots()

    fig3, ax3 = plt.subplots()

    mass_ratio_list = []

    for wid, merger_info in all_mergers.items():

        for nth in range(len(merger_info['times'])):
            
            time = merger_info['times'][nth]

            host_mass = merger_info['host_masses'][nth]

            partner_mass = merger_info['partner_masses'][nth]

            merger_type = merger_info['merger_types'][nth]

            if 'HG' in merger_type:
                color = 'red'
            elif ('A' in merger_type) or ('C' in merger_type):
                color = 'blue'
            else:
                color = 'black'

            chirp_mass = calc_chirp_mass(host_mass, partner_mass)
            
            if host_mass > partner_mass:
                mass_ratio = partner_mass/host_mass
            else:
                mass_ratio = host_mass/partner_mass

            mass_ratio_list.append(mass_ratio)

            ax.scatter(time, chirp_mass, color=color, s=5)
            ax2.scatter(chirp_mass, mass_ratio, color=color, s=5)

            if 'E' in merger_type:
                ax.scatter(time, chirp_mass, facecolors='none',
                           edgecolors='pink')
                ax2.scatter(chirp_mass, mass_ratio, facecolors='none',
                            edgecolors='pink')

    ax.scatter([], [], color='red', label="Higher Gen")
    ax.scatter([], [], color='blue', label="1G special")
    ax.scatter([], [], color='black', label="1G")
    ax.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')
    ax.set_xlabel('time (code unit)')
    ax.set_ylabel(r'chirp mass ($M_\odot$)')
    ax.legend(fontsize=8)

    ax2.scatter([], [], color='red', label="Higher Gen")
    ax2.scatter([], [], color='blue', label="1G special")
    ax2.scatter([], [], color='black', label="1G")
    ax2.scatter([], [], facecolors='none', edgecolors='pink', 
               label='Escape merger')
    ax2.set_xlabel(r'chirp mass ($M_\odot$)')
    ax2.set_ylabel('mass ratio')
    ax2.legend(fontsize=8)

    # Create histogram
    hist, bin_edges = np.histogram(mass_ratio_list, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian function
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Fit
    popt, _ = scp.curve_fit(gaussian, bin_centers, hist)

    ax3.hist(mass_ratio_list, bins=50, density=True, alpha=0.6)
    ax3.plot(
        bin_centers, gaussian(bin_centers, *popt), 'r-', lw=2,
        label="Median : " + str(np.round(np.median(mass_ratio_list),2))
        )

    ax3.set_xlabel("mass ratio")
    ax3.set_ylabel("Number per mass ratio")
    ax3.legend()

    fig.savefig('merger_plot.pdf')
    fig2.savefig('merger_plot2.pdf')
    fig3.savefig('mass_ratio_peak.pdf')

    return ax