import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
from IPython.display import clear_output

def _bootstrap_error(data, function, num_bs=100):
    assert data.ndim == 1, '_bootstrap_error: wrong data dimention'
    bs_data = np.random.choice(data, size=(num_bs, len(data)), replace=True)
    return np.array([function(bs_sample) for bs_sample in bs_data]).std()


def _get_stats(arr):
    class Obj:
        pass

    result = Obj()

    result.mean = arr.mean()
    result.width = arr.std()

    result.mean_err = result.width / (len(arr) - 1) ** 0.5
    result.width_err = _bootstrap_error(arr, np.std)

    return result


def compare_two_dists(d_real, d_gen, label, tag=None, nbins=100):
    ax = plt.gca()
    bins = np.linspace(min(d_real.min(), d_gen.min()), max(d_real.max(), d_gen.max()), nbins + 1)

    stats_real = _get_stats(d_real)
    stats_gen = _get_stats(d_gen)

    if tag:
        leg_entry = f'gen ({tag})'
    else:
        leg_entry = 'gen'

    plt.hist(d_real, bins=bins, density=True, label='real')
    plt.hist(d_gen, bins=bins, density=True, label=leg_entry, histtype='step', linewidth=2.0)

    string = '\n'.join(
        [
            f"real: mean = {stats_real.mean :.4f} +/- {stats_real.mean_err :.4f}",
            f"gen:  mean = {stats_gen.mean :.4f} +/- {stats_gen .mean_err :.4f}",
            f"real: std  = {stats_real.width:.4f} +/- {stats_real.width_err:.4f}",
            f"gen:  std  = {stats_gen.width:.4f} +/- {stats_gen .width_err:.4f}",
        ]
    )
    default_family = rcParams['font.family']
    rcParams['font.family'] = 'monospace'
    ax.add_artist(AnchoredText(string, loc=2))
    rcParams['font.family'] = default_family

    plt.xlabel(label)
    plt.legend()


def plot_metrics(gen_loss, disc_loss):
    clear_output()
    fig=plt.figure(figsize=(16, 8))
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    lns1 = ax.plot(gen_loss, color="C0", label="generator")
    ax.set_xlabel("step", color="C0")
    ax.set_ylabel("loss", color="C0")

    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    lns2 = ax2.plot(disc_loss, color="C1", label="discriminator")
    ax2.xaxis.tick_top()
    ax2.set_xlabel('epoch', color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")
    ax2.set_yticks([])
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    plt.show()
