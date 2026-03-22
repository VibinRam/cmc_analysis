import matplotlib.pyplot as plt
from cmc_analysis.utils import *

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

            ax.scatter(time, chirp_mass, color=color, s=5)

    ax.scatter([], [], color='red', label="Higher Gen")
    ax.scatter([], [], color='blue', label="1G special")
    ax.scatter([], [], color='black', label="1G")
    ax.set_xlabel('time (code unit)')
    ax.set_ylabel(r'chirp mass ($M_\odot$)')
    ax.legend(fontsize=8)

    fig.savefig('merger_plot.pdf')

    return ax