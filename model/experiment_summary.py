import sys
import json
import cufflinks

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from classes.dataset.Generator import *
from classes.diego_sampler import *

argv = sys.argv[1:]


def open_experiment_data_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def parse_experiment_data(metrics_as_df):
    epochs = []
    train_loss = []
    train_levenshtein = []
    train_acc = []
    val_loss = []
    val_levenshtein = []
    val_acc = []
    for epoch, row in metrics_as_df.iterrows():
        epochs.append(epoch)
        train_loss.append(row['train_loss'])
        train_levenshtein.append(row['train_lev_distance'])
        train_acc.append(row['train_accuracy'])
        val_loss.append(row['val_loss'])
        val_levenshtein.append(row['val_lev_distance'])
        val_acc.append(row['val_accuracy'])

    return epochs, (train_loss, train_levenshtein, train_acc),\
        (val_loss, val_levenshtein, val_acc)


def plot_loss_function(loss_json_file, epoch_start_offset):
    epoch_start_offset = int(epoch_start_offset)
    loss_file_name = 'loss_outputs/{}'.format(loss_json_file)
    metrics_as_dict = open_experiment_data_file(loss_file_name)
    metrics_df = pd.DataFrame(metrics_as_dict)
    epochs, loss, levenshtein = parse_experiment_data(metrics_df)
    epochs, loss, levenshtein = epochs[epoch_start_offset:], loss[epoch_start_offset:], levenshtein[epoch_start_offset:]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('crossentropy', color=color)
    ax1.plot(epochs, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Levenshtein distance',
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, levenshtein, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('plots/metrics_experiment_{}_{}_epochs.png'.
                format(loss_json_file.replace(".json", ""), epoch_start_offset),
                bbox_inches="tight")
    return


def create_model_metrics_plot(loss_json_file, epoch_start_offset=0):
    epoch_start_offset = int(epoch_start_offset)
    loss_file_name = 'loss_outputs/{}'.format(loss_json_file)

    metrics_as_dict = open_experiment_data_file(loss_file_name)
    metrics_df = pd.DataFrame(metrics_as_dict)
    epochs, train_metrics, val_metrics = parse_experiment_data(metrics_df)

    train_loss, train_levenshtein, train_acc = train_metrics
    val_loss, val_levenshtein, val_acc = val_metrics
    epochs, train_loss, train_levenshtein, train_acc =\
        epochs[epoch_start_offset:], train_loss[epoch_start_offset:],\
        train_levenshtein[epoch_start_offset:], train_acc[epoch_start_offset:]
    val_loss, val_levenshtein, val_acc = \
        val_loss[epoch_start_offset:], val_levenshtein[epoch_start_offset:],\
        val_acc[epoch_start_offset:]

    fig, axes = plt.subplots(3, 2, sharey='row')
    axes[0,0].set_title('Entrenamiento')
    axes[0,1].set_title('Prueba')
    axes[0,0].plot(epochs, train_loss)
    axes[0,0].set_ylabel('Entropia Cruzada')
    axes[0,1].plot(epochs, val_loss)
    axes[1,0].plot(epochs, train_acc)
    axes[1,0].set_ylabel('Precision')
    axes[1,1].plot(epochs, val_acc)
    axes[2,0].plot(epochs, train_levenshtein)
    axes[2,0].set_xlabel('Epochs')
    axes[2,0].set_ylabel('Levenshtein')
    axes[2,1].plot(epochs, val_levenshtein)
    axes[2,1].set_xlabel('Epochs')

    model_name = r"""con atenci$\'o$n""" if "attention" in loss_json_file \
        else "poco profundo"
    if ("attention" not in loss_json_file) and ("shallow" not in loss_json_file):
        model_name = "original"

    fig.suptitle('Metricas modelo {}'.format(model_name), fontsize=16)

    plt.savefig('plots/metrics_experiment_{}_{}_epochs.png'.
                format(loss_json_file.replace(".json", ""), epoch_start_offset),
                bbox_inches="tight")
    return


loss_json_file = argv[0]
epoch_start_offset = argv[1]
create_model_metrics_plot(loss_json_file, epoch_start_offset)
print("created")
