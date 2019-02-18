import sys
import json
import cufflinks

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from keras.models import load_model

from classes.dataset.Generator import *
from classes.diego_sampler import *

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
argv = sys.argv[1:]


def open_experiment_data_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def parse_experiment_data(metrics_as_df):
    epochs = []
    loss = []
    levenshtein = []
    for epoch, row in metrics_as_df.iterrows():
        epochs.append(epoch)
        loss.append(row['loss'])
        levenshtein.append(row['lev_distance'])
    return epochs, loss, levenshtein


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
    plt.savefig('plots/metrics_experiment_{}.png'.
                format(loss_json_file.replace(".json", "")),
                bbox_inches="tight")


loss_json_file = argv[0]
epoch_start_offset = argv[1]
plot_loss_function(loss_json_file, epoch_start_offset)
print("ran all")

# model = load_model(trained_model_name)
# visual_input_shape = model.inputs[0].shape.as_list()
# sequence_input_shape = model.inputs[1].shape.as_list()
#
# sampler = Sampler(voc_path='../bin', output_size=output_size,
#                   context_length=CONTEXT_LENGTH)


