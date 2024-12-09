# code copied from: https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py

import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_value_function(V, title="Value Function",
                        directory = None, file_name = None,
                        show = False):
    """
    Plots the value function as a surface plot.

    code copied from: https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2,
                                  np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2,
                                np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title, usable_ace):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.5)
        ax.set_xlabel('Player Sum', size=22)
        ax.set_ylabel('Dealer Showing', size=22)
        ax.set_zlabel('Value', size=22)
        ax.set_title(title, size=22)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        if file_name:
            fig.savefig(directory + usable_ace + file_name, bbox_inches = 'tight')
        if show:
            fig.show()
        plt.close(fig)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title),
                 usable_ace = "noace_")
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title),
                 usable_ace = "ace_")


def plot_avg_reward_episode(path, env_types, ndecks):
    """
    Function which plots the average return over episodes

    path: path to the folder where the data resides
    env_types: list with the env types you want to plot
    ndecks: a list with the decks that you want to plot
    save_path: optional path of where to save the fig.
    """
    def load_df(path, env_type, ndecks):
        assert env_type in ["hand_MC", "hand", "sum"]
        path_to_file = "{}/{}_state_{}.txt".format(path, env_type, ndecks)
        df = pd.read_table(path_to_file, sep=",")
        df['env_type'] = env_type
        df['ndecks'] = ndecks
        return df

    df_l = []
    for env in env_types:
        for deck in ndecks:
            df_l.append(load_df(path, env, deck))
    df = pd.concat(df_l)

    fig, ax = plt.subplots(figsize=(8,6))
    lab= []
    for label, df in df.groupby(["env_type", "ndecks"]):
        lab.append(label)
        ax.plot(df['episode'], df['avg_reward'], label=label)

    lgd = ax.legend(title="(State space, ndecks)", loc='upper center',
                    bbox_to_anchor=(0.5, -0.1),
              shadow=False, ncol=2, framealpha=0.0, fontsize=22)
    ax.set_xlabel("episode", size=22)
    ax.set_ylabel("avg. reward", size=22)
    return fig, lgd



def plot_policy(Q, title="Policy Function",
                        directory = None, file_name = None,
                        show = False):
    """
    Plots the policy (best actions) as a heatmap.
    Q should be a dictionary mapping state tuples to arrays of action values.
    Colors indicate action: blue for stick, red for hit
    """
    # Set a professional matplotlib style
    plt.style.use('seaborn-v0_8-white')
    
    min_x = min(k[0] for k in Q.keys())
    max_x = max(k[0] for k in Q.keys())
    min_y = min(k[1] for k in Q.keys())
    max_y = max(k[1] for k in Q.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(1, max_y + 1)  # Start dealer showing from 1
    X, Y = np.meshgrid(x_range, y_range)

    # Find best action for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: np.argmax(Q[(_[0], _[1], False)]), 2,
                                  np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: np.argmax(Q[(_[0], _[1], True)]), 2,
                                np.dstack([X, Y]))

    def plot_heatmap(Z, title, usable_ace):
        # Create figure with more whitespace and professional look
        fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
        
        # Adjust subplot margins
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        
        # Custom color palette with softer colors
        cmap = matplotlib.colors.ListedColormap(['#3498db', '#e74c3c'])  # Softer blue and red
        bounds = [0, 0.5, 1]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create heatmap with slight padding
        im = ax.imshow(Z, cmap=cmap, norm=norm, aspect='auto',
                      extent=[min_x-0.5, max_x+0.5, max_y+0.5, 0.5],
                      interpolation='nearest')
        
        # Styling grid and ticks
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xticklabels(x_range, fontsize=10)
        ax.set_yticklabels(y_range, fontsize=10)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Professional labels and title
        ax.set_xlabel('Player Sum', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dealer Showing', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Create a custom legend instead of a colorbar
        import matplotlib.patches as mpatches
        stick_patch = mpatches.Patch(color='#3498db', label='Stick')
        hit_patch = mpatches.Patch(color='#e74c3c', label='Hit')
        ax.legend(handles=[stick_patch, hit_patch], 
                  title='Action', 
                  loc='center left', 
                  bbox_to_anchor=(1.05, 0.5),
                  title_fontsize=12,
                  fontsize=10)
        
        if file_name:
            fig.savefig(directory + usable_ace + file_name, 
                        bbox_inches='tight', 
                        dpi=300)  # High-resolution save
        if show:
            fig.show()
        plt.close(fig)

    plot_heatmap(Z_noace, "{} (No Usable Ace)".format(title),
                 usable_ace = "noace_")
    plot_heatmap(Z_ace, "{} (Usable Ace)".format(title),
                 usable_ace = "ace_")
