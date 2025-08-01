"""
Plotting utilities for training visualization.
"""
import matplotlib.pyplot as plt
import os


def plot_training_progress(sum_rewards, agent_name, save_folder='saved_plots', 
                         title='Progress', xlabel='Episodes', 
                         ylabel='Aggregate Reward (Past 200 Episodes)'):

    plt.figure()
    plt.plot(sum_rewards)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if agent_name:
        filename = agent_name + '.png'
    else:
        filename = 'training_plot.png'
    
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    
    return save_path
