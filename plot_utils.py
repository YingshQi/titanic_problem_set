from cProfile import label
import matplotlib.pyplot as plt
import seaborn as sns

COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]


def plot_count_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette=COLOR_LIST)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title}")
    plt.show()


def plot_distribution_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(data_df[hue].unique()):
        g = sns.histplot(
            data_df.loc[data_df[hue] == h, feature], color=COLOR_LIST[i], ax=ax, label=h
        )
    ax.set_title(f"Number of passengers / {title}")
    g.legend()
    plt.show()
