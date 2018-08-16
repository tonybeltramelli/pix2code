import sys
import json
import cufflinks

import plotly.offline as po
import pandas as pd
import tensorflow as tf


from keras.models import load_model

from classes.dataset.Generator import *
from classes.diego_sampler import *

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
argv = sys.argv[1:]


def plot_loss_function(model_name, data_percentage, loss_function_name='CrossEntropy'):
    experiment_name = '{}_{}_model'.format(model_name, data_percentage)
    loss_file_name = 'loss_outputs/loss_history_{}_model.json'.format(experiment_name)

    with open(loss_file_name, 'r') as file:
        loss = json.load(file)
    loss = pd.DataFrame(loss)
    layout = dict(xaxis=dict(title='epochs'), title='Loss through epochs',
                  yaxis=dict(title=loss_function_name))
    fig = loss.figure(layout=layout)
    po.plot(fig, filename='plots/loss_{}.png'.format(experiment_name), auto_open=False)


if len(argv) < 3:
    print("Error: not enough argument supplied:")
    print("generate.py <trained model name> <input image> <output path> <search method (default: greedy)>")
    exit(0)
else:
    model_name = argv[0]
    input_path = argv[1]
    output_path = argv[2]
    search_method = argv[3] or 'greedy'
    data_percentage = argv[4]
    plot_loss_function(model_name, data_percentage)

# model = load_model(trained_model_name)
# visual_input_shape = model.inputs[0].shape.as_list()
# sequence_input_shape = model.inputs[1].shape.as_list()
#
# sampler = Sampler(voc_path='../bin', output_size=output_size,
#                   context_length=CONTEXT_LENGTH)


