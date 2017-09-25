from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from .Vocabulary import *
from .BeamSearch import *
from .Utils import *


class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            probas = model.predict(input_img, np.array([current_context]))
            prediction = np.argmax(probas)
            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def recursive_beam_search(self, model, input_img, current_context, beam, current_node, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))

        predictions = []
        for i in range(0, len(probas)):
            predictions.append((i, probas[i], probas))

        nodes = []
        for i in range(0, len(predictions)):
            prediction = predictions[i][0]
            score = predictions[i][1]
            output_probas = predictions[i][2]
            nodes.append(Node(prediction, score, output_probas))

        beam.add_nodes(current_node, nodes)

        if beam.is_valid():
            beam.prune_leaves()
            if sequence_length == 1 or self.voc.token_lookup[beam.root.max_child().key] == END_TOKEN:
                return

            for node in beam.get_leaves():
                prediction = node.key

                new_context = []
                for j in range(1, self.context_length):
                    new_context.append(current_context[j])
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)

                self.recursive_beam_search(model, input_img, new_context, beam, node, sequence_length - 1)

    def predict_beam_search(self, model, input_img, beam_width=3, require_sparse_label=True, sequence_length=150):
        predictions = START_TOKEN
        out_probas = []

        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        beam = BeamSearch(beam_width=beam_width)

        self.recursive_beam_search(model, input_img, current_context, beam, beam.root, sequence_length)

        predicted_sequence, probas_sequence = beam.search()

        for k in range(0, len(predicted_sequence)):
            prediction = predicted_sequence[k]
            probas = probas_sequence[k]
            out_probas.append(probas)

            predictions += self.voc.token_lookup[prediction]

        return predictions, out_probas
