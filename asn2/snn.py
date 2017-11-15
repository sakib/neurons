""" Spiking Neural Network implementation """
import numpy as np
from pprint import pprint as pp
from collections import defaultdict
from itertools import tee, izip, product
from neurons import LIF as Neuron

CALIBRATION_INFO = False

def pairwise(iterable):
    """ Function to obtain in-order tuples on a list.

    Args:
        iterable (:obj:`list`): Any iterable on a collection. For example:
            [s0, s1, s2, s3, ...]

    Returns:
        :obj:`list` of :obj:`tuple`: Pairwise tuples from input collection:
            [(s0, s1), (s1, s2), (s2, s3), ...]

    """
    a, b = tee(iterable)
    next(b)
    return zip(a, b)


class SNN(object):
    """Implementation of Spiking Neural Network.

    Uses temporal encoding of inputs, specifically the time-to-first-spike.

    Attributes:
        layers (:obj:`list` of :obj:`LIF`): Layers of leaky-IF neurons.
        synapses (:obj:`list` of :obj:`np.ndarray`): Matrices of weights where
            rows correspond to presynaptic and cols correspond to postsynaptic.
    """
    def __init__(self, n_layers):
        """Initialize an SNN with random weights in range [-1, 1].

        Args:
            n_layers (:obj:`list` of :obj:`int`): Each entry is a number which
                corresponds to the number of neurons in that layer.
        """
        self.layers = [[Neuron() for i in range(n)] for n in n_layers]
        self.synapses = [np.random.rand(len1, len2)*2-1
                         for len1, len2 in pairwise(n_layers)]

    def classify(self, problem, inputs):
        """For a specific problem, train + run the SNN til output neuron fires.

        Args:
            problem (str): Specifies the problem name. For example: 'XOR'
            inputs (:obj:`list` of :obj:`str`): Has one input per input neuron.

        Returns:
            Output of the network for the particular `problem` and `inputs`.

        """
        if problem == 'XOR':
            # TODO: train_xor(inputs)
            return self.classify_xor(inputs)

        return None

    def classify_xor(self, inputs):
        """Solve the XOR problem.

        Input1  Input2  Output
        0       0       0
        0       1       1
        1       0       1
        1       1       0

        Args:
            inputs (:obj:`list` of :obj:`str`): Two values, both '0' or '1'.

        Returns:
            Currently, the time-to-first-spike of the output neuron.
            TODO: Figure out how to train network to calibrate + output 0 or 1.

        """
        time_threshold = 101
        inp_to_curr = defaultdict(int)
        inp_to_curr[0] = self.calibrate_current(6)
        inp_to_curr[1] = self.calibrate_current(3)

        for c_time in range(time_threshold):
            for i_c_layer in range(len(self.layers)):
                for i_c_neuron in range(len(self.layers[i_c_layer])):
                    neuron = self.layers[i_c_layer][i_c_neuron]
                    if neuron.last_spike == -1: # not yet spiked
                        if i_c_layer == 0: curr = inp_to_curr[inputs[i_c_neuron]]
                        else: curr = self.sum_epsps(c_time, i_c_layer, i_c_neuron)
                        neuron.time_step(curr, c_time)

        return self.layers[-1][0].last_spike # output neuron ttfs

    def sum_epsps(self, c_time, i_c_layer, i_c_neuron):
        """Given active SNN and non-input neuron, obtain sum of input currents.

        Note:
            This is a helper method meant to make `classify_xor()` more sensible
            and will only yield a meaningful (nonzero) current value when one or
            more of the given neuron's presynaptic neurons have spiked already,
            and when `c_time` exceeds these `p_spike` spike times.

        Args:
            c_time (int): Current time of the running network. Must exceed
                presynaptic neurons' spike times.
            i_c_layer (int): Index of the layer of the given neuron in the SNN.
            i_c_neuron (int): Index of the given neuron in its particular layer.

        Returns:
            The value of the input current that should be fed into the neuron.

        """
        current = 0
        c_layer = self.layers[i_c_layer]
        p_layer = self.layers[i_c_layer-1]
        c_synapses = self.synapses[i_c_layer-1]
        for i_p_neuron in range(len(p_layer)):
            p_neuron = p_layer[i_p_neuron]
            p_spike = p_neuron.last_spike
            if p_spike != -1: # ttfs exists
                weight = c_synapses[i_p_neuron][i_c_neuron]
                current += weight * 2**-(c_time-p_spike)
        return current


    def calibrate_current(self, ttfs):
        """Get the electric current constant needed to make neuron spike first.

        Args:
            ttfs (int): Target time-to-first-spike for the neuron.

        Returns:
            Value of constant current for which, if you supplied it to the
            default neuron, would make it spike first at time `ttfs`.

        """
        for current in range(0, 10000, 1):
            neuron = Neuron()
            for time in range(0, ttfs+1):
                neuron.time_step(current/10, time)
                if neuron.last_spike != -1: break
            if neuron.last_spike == ttfs:
                return current/10


if __name__ == '__main__':
    snn = SNN([2, 4, 1])
    problems = defaultdict()
    problems['XOR'] = [0, 1]

    if CALIBRATION_INFO:
        print('Neuron values:')
        pp(vars(Neuron()))
        print('')

        for ttfs in range(1, 21, 1):
            print('ttfs: {}, current: {}'.format(
                ttfs, snn.calibrate_current(ttfs)))

    for name, inputs in problems.items():
        print('\n{}\n{}'.format(name, '-'*len(name)))
        for inp in list(product(inputs, inputs)): # all inputs
            print('{}: {}'.format(inp, snn.classify(name, inp)))

