import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

cm = 1 / 2.54
mm = 1 / 25.4


def adjust_alpha(color, amount=0.5):
    """
    Lightens (amount < 1) or darkens (amount > 1) the given color.

    Parameters
    ----------
    color : str or tuple
        Matplotlib color name or a tuple (r, g, b).
    amount : float
        Factor by which to adjust the color’s lightness.
        - Values between 0 and 1 will lighten the color
          (0 returns white).
        - Values > 1 will darken the color.

    Returns
    -------
    new_color : tuple
        The adjusted color as an RGB tuple.
    """
    try:
        # If 'color' is a named color like 'red', convert to hex
        c = mc.cnames[color]
    except KeyError:
        # Otherwise, assume it's already a valid hex or RGB
        c = color

    # Convert to RGB in [0, 1], then to HLS
    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Adjust lightness
    # if amount < 1, we move l closer to 1 (white) -> lighten
    # if amount > 1, we move l closer to 0 (black) -> darken
    new_l = l + (1 - l) * (1 - amount) if amount < 1 else l * (2 - amount)

    # Convert back to RGB
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
    return (new_r, new_g, new_b)


def set_base_style():
    plt.rcParams.update(
        {
            "font.sans-serif": "Noto Sans",
            "font.family": "sans-serif",
            # "text.usetex": True,
            # "text.latex.preamble": r"\usepackage{wasysym} \usepackage{siunitx} \sisetup{detect-all} \usepackage{sansmath} \sansmath \usepackage[sfdefault]{noto} \usepackage[T1]{fontenc}",
            "axes.titlelocation": "left",
            "axes.titlesize": "medium",
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            # "figure.dpi": 100,
            "image.origin": "lower",
            "savefig.pad_inches": 0.0,
            # Font sizes smaller
            # "font.size": 8,
        }
    )


def set_light_style():
    set_base_style()
    plt.rcParams.update(
        {
            "lines.markeredgecolor": "k",
            "lines.markeredgewidth": 5,
            "patch.facecolor": "#000000",
            "patch.edgecolor": "#000000",
            "boxplot.flierprops.color": "#000000",
            "boxplot.flierprops.markeredgecolor": "#000000",
            "boxplot.boxprops.color": "#000000",
            "boxplot.whiskerprops.color": "#000000",
            "boxplot.capprops.color": "#000000",
            "boxplot.medianprops.color": "#000000",
            "boxplot.meanprops.color": "#000000",
            "boxplot.meanprops.markerfacecolor": "#000000",
            "boxplot.meanprops.markeredgecolor": "#000000",
            "text.color": "#000000",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ],
            ),
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "grid.color": "#7f7f7f",
            "figure.facecolor": "#FFFFFF",
            "figure.edgecolor": "#000000",
            "image.cmap": "viridis",
        }
    )


def set_dark_style():
    set_base_style()
    plt.rcParams.update(
        {
            "patch.facecolor": "#FFFFFF",
            "patch.edgecolor": "#FFFFFF",
            "boxplot.flierprops.color": "#FFFFFF",
            "boxplot.flierprops.markeredgecolor": "#FFFFFF",
            "boxplot.boxprops.color": "#FFFFFF",
            "boxplot.whiskerprops.color": "#FFFFFF",
            "boxplot.capprops.color": "#FFFFFF",
            "boxplot.medianprops.color": "#FFFFFF",
            "boxplot.meanprops.color": "#FFFFFF",
            "boxplot.meanprops.markerfacecolor": "#FFFFFF",
            "boxplot.meanprops.markeredgecolor": "#FFFFFF",
            "text.color": "#FFFFFF",
            "axes.facecolor": "#000000",
            "axes.edgecolor": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ],
            ),
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
            "grid.color": "#7f7f7f",
            "figure.facecolor": "#000000",
            "figure.edgecolor": "#FFFFFF",
            "image.cmap": "magma",
        }
    )


def get_ylims(cls, ydata):
    allfunds_tmp = np.concatenate(ydata).ravel().tolist()
    lower = np.min(allfunds_tmp)
    upper = np.max(allfunds_tmp)
    return lower, upper


def transparent_fade_colormap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def hide_ax(ax):
    ax.xaxis.set_visible(False)
    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(left=False, labelleft=False)
    ax.patch.set_visible(False)


def hide_xax(ax):
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)


def hide_yax(ax):
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)


def set_boxplot_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color="black")


def circle_annotate(ax, xy, xy_adjust_text, text):
    xy = np.array(xy)
    xy_adjust_text = np.array(xy_adjust_text)
    ax.text(
        *xy,
        " ",
        ha="center",
        va="center",
        color="black",
        zorder=1100,
        transform=ax.transData,
        clip_on=False,
        bbox=dict(
            boxstyle="circle,pad=0.1",
            fc="w",
            ec="k",
            lw=0.5,
            alpha=1,
        ),
        fontsize="small",
    )

    xy_text = xy + xy_adjust_text
    ax.text(
        *xy_text,
        text,
        ha="center",
        va="center_baseline",
        color="black",
        zorder=1101,
        transform=ax.transData,
        clip_on=False,
        fontsize="small",
    )


def letter_subplots(
    axes=None,
    letters=None,
    xoffset=-0.1,
    yoffset=1.0,
    **kwargs,
):
    """Add letters to the corners of subplots (panels). By default each axis is
    given an uppercase bold letter label placed in the upper-left corner.
    Args
        axes : list of pyplot ax objects. default plt.gcf().axes.
        letters : list of strings to use as labels, default ["A", "B", "C", ...]
        xoffset, yoffset : positions of each label relative to plot frame
        (default -0.1,1.0 = upper left margin). Can also be a list of
        offsets, in which case it should be the same length as the number of
        axes.
        Other keyword arguments will be passed to annotate() when panel letters
        are added.

    Returns
    -------
        list of strings for each label added to the axes
    Examples:
        Defaults:
            >>> fig, axes = plt.subplots(1,3)
            >>> letter_subplots() # boldfaced A, B, C

        Common labeling schemes inferred from the first letter:
            >>> fig, axes = plt.subplots(1,4)
            # panels labeled (a), (b), (c), (d)
            >>> letter_subplots(letters='(a)')
        Fully custom lettering:
            >>> fig, axes = plt.subplots(2,1)
            >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
        Per-axis offsets:
            >>> fig, axes = plt.subplots(1,2)
            >>> letter_subplots(axes, xoffset=[-0.1, -0.15])

        Matrix of axes:
            >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
            # fig.axes is a list when axes is a 2x2 matrix
            >>> letter_subplots(fig.axes)
    """
    # get axes:
    if axes is None:
        axes = plt.gcf().axes
    # handle single axes:
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(axes)])
    llets = list("abcdefghijklmnopqrstuvwxyz"[: len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [f"({lett})" for lett in llets]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [f"({lett})" for lett in ulets]
        fontweight = "normal"
    elif letters in ("lower", "lowercase", "a"):
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset] * len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset] * len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(
        fontweight=fontweight,
        fontsize="large",
        ha="center",
        va="center",
        xycoords="axes fraction",
        annotation_clip=False,
    )
    kwargs = dict(list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax, lbl, xoff, yoff in zip(
        axes, letters, xoffset, yoffset, strict=False
    ):
        t = ax.annotate(lbl, xy=(xoff, yoff), **kwargs)
        list_txts.append(t)

    return list_txts


def float_rgb_to_hex(rgb):
    return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"


def show_palette(colors):
    n_colors = len(colors)
    plt.figure(figsize=(n_colors, 1))
    plt.imshow([colors], extent=[0, n_colors, 0, 1], aspect="auto")
    plt.axis("off")
    plt.show()


def example_plot():
    figsize = (16 * cm, 8 * cm)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1])
    ax.scatter([0, 1], [0, 1], color="r", s=10)
    circle_annotate(ax, (0.5, 0.5), (0, 0), "1")
    letter_subplots(ax)
    ax.set_title("Example Plot")
    plt.show()


def main():
    set_light_style()
    example_plot()
    set_dark_style()
    example_plot()


loser_color = adjust_alpha("#236477", 0.85)
winner_color = adjust_alpha("#4a6c2f", 0.85)
baseline_color = "grey"
other_color = "#595EC5"
# loser_color = "#2C828C"
# winner_color = "#4F9260"

cmap1_ = sns.color_palette("crest_r", as_cmap=True)
cmap2_ = sns.color_palette("flare", as_cmap=True)
loser_color = cmap1_(0.5)
winner_color = adjust_alpha(cmap1_(0.9), 1.2)
other_color = cmap1_(0.05)


if __name__ == "__main__":
    main()
    print(f"Winner color: {float_rgb_to_hex(winner_color)}")
    print(f"Loser color: {float_rgb_to_hex(loser_color)}")
