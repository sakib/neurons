""" Spiking Neural Network implementation """
import math
import numpy as np
from pprint import pprint as pp
from collections import defaultdict
from itertools import tee, izip, product
from sklearn.preprocessing import normalize
from neurons import LIF as Neuron

VERBOSE = False

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
        time_const (int): Time constant for the SNN. Smaller is faster.
        learn_rate (int): Learning rate for the SNN. Smaller is faster.
        layers (:obj:`list` of :obj:`LIF`): Layers of leaky-IF neurons.
        trained (:obj:`dict` of :obj:`str`): Records whether SNN is trained
            on a given problem, for example, 'XOR'.
        synapses (:obj:`list` of :obj:`np.ndarray`): Matrices of weights where
            rows correspond to presynaptic and cols correspond to postsynaptic.
    """
    def __init__(self, n_layers):
        """Initialize an SNN with random weights in range [0, 1].

        Args:
            n_layers (:obj:`list` of :obj:`int`): Each entry is a number which
                corresponds to the number of neurons in that layer.
        """
        self.time_const = 3.
        self.learn_rate = 7.
        self.excit_ratio = 2
        self.trained = defaultdict()
        self.layers = [[Neuron() for i in range(n)] for n in n_layers]
        self.synapses = [np.random.rand(len1, len2) for len1, len2 in pairwise(n_layers)]
        # make half of hidden layer inhibitory
        mid, hid_out = list(map(lambda l: l[len(self.layers)/2], [self.layers, self.synapses]))
        for i in range(int(math.ceil(len(mid)/self.excit_ratio)), len(mid)):
            mid[i].type = 'inhibitory' # invert inhib weights
            for j in range(len(hid_out[i])): hid_out[i][j] *= -1


    def train(self, problem, inp_maps, out_maps):
        """For a specific problem, train the SNN.

        Args:
            problem (str): Specifies the problem name. For example: 'XOR'
            inp_maps (:obj:`list` of int): Has one time-to-first-spike per input.
            out_maps (:obj:`list` of int): Has one time-to-first-spike per output.

        """
        if not problem in self.trained:
            if problem == 'XOR':
                self.reset()
                self.train_xor(inp_maps, out_maps)
                self.reset()
            self.trained[problem] = True


    def train_xor(self, inp_maps, out_maps):
        """Train the SNN to solve the XOR problem.

        Args:
            inp_maps (:obj:`list` of `int`): Two values, input ttfs for 0 and 1.
            out_maps (:obj:`list` of `int`): Two values, output ttfs for 0 and 1.

        """
        teacher = Neuron()
        t_window, t_threshold = 5, 1000
        output = self.layers[-1][0] # output neuron
        all_inputs = map(lambda ttfs_pair: [self.calibrate_current(ttfs)
                for ttfs in ttfs_pair], list(product(inp_maps, inp_maps)))

        for input_currs in all_inputs:
            self.reset()
            print('\ntraining {} in the hyperbolic time chamber...'.format(input_currs))
            target_ttfs = out_maps[0] if input_currs[0] == input_currs[1] else out_maps[1]
            #for x in range(2):
            for c_time in range(1, t_threshold):
                #print('the time is {}ms'.format(c_time))
                for i_c_layer in range(len(self.layers)):
                    for i_c_neuron in range(len(self.layers[i_c_layer])):
                        neuron = self.layers[i_c_layer][i_c_neuron]
                        if i_c_layer == 0: curr = input_currs[i_c_neuron]
                        else: curr = self.sum_epsps(c_time, i_c_layer, i_c_neuron)
                        if neuron is output: curr += 1000 if c_time % target_ttfs == 0 else -1000
                        neuron.time_step(curr, c_time)
                        #if neuron is output and neuron.last_spike == c_time: print('\toutput spikes!')
                        #elif neuron.last_spike == c_time: print('\tlayer {} neuron {} type {} spikes!'.format(i_c_layer, i_c_neuron, neuron.type))
                if c_time % target_ttfs == 0:
                    self.update_weights(c_time, t_window)
                    #print('the weights are being updated!')
                    #for synapse in self.synapses:
                    #    for row in synapse: print('\t{}'.format(row))
                    break
            #print('(0,0): {}'.format(self.classify_xor([0, 0], inp_maps, out_maps)))
            #print('(0,1): {}'.format(self.classify_xor([0, 1], inp_maps, out_maps)))
            #print('(1,0): {}'.format(self.classify_xor([1, 0], inp_maps, out_maps)))
            #print('(1,1): {}'.format(self.classify_xor([1, 1], inp_maps, out_maps)))

        print('Trained!')


    def update_weights(self, c_time, t_window):
        """Update weights of all synapses in the SNN.

        Args:
            c_time (int): Current timestep in the SNN.
            t_window (int): Length of the time window in the SNN to check for spikes in.

        """

        ch = 0.2

        def has_spiked(neuron, c_time=c_time, t_window=t_window):
            return neuron.last_spike > max(-1, c_time - t_window)

        for i_c_synapse in range(len(self.synapses)):
            c_synapse = self.synapses[i_c_synapse]
            pre_layer, post_layer = self.layers[i_c_synapse:i_c_synapse+2]
            for post_neuron in post_layer:
                for pre_neuron in pre_layer:
                    r, c = pre_layer.index(pre_neuron), post_layer.index(post_neuron)
                    if pre_neuron.type == 'excitatory':
                        if has_spiked(pre_neuron):
                            delta = (1-c_synapse[r][c])*1.0/self.learn_rate - ch*c_synapse[r][c]
                            if has_spiked(post_neuron): c_synapse[r][c] += delta # both spiked
                            else: c_synapse[r][c] -= delta # pre spiked, post didn't
                        else:
                            c_synapse[r][c] -= ch*c_synapse[r][c]
                    elif pre_neuron.type == 'inhibitory':
                        if has_spiked(pre_neuron):
                            delta = (1+c_synapse[r][c])*1.0/self.learn_rate - ch*c_synapse[r][c]
                            if not has_spiked(post_neuron): c_synapse[r][c] -= delta # choke
                            else: c_synapse[r][c] += delta # both spiked
                        else:
                            c_synapse[r][c] += ch*c_synapse[r][c]

            """# column normalization
            col_sums = defaultdict(int)
            for r in range(len(c_synapse)):
                for c in range(len(c_synapse[r])):
                    col_sums[c] += c_synapse[r][c]
            for r in range(len(c_synapse)):
                for c in range(len(c_synapse[r])):
                    c_synapse[r][c] /= col_sums[c] """


    def classify(self, problem, inputs, inp_maps, out_maps):
        """For a specific problem, run the SNN until output neuron fires.

        Args:
            problem (str): Specifies the problem name. For example: 'XOR'
            inputs (:obj:`list` of :obj:`int`): Has one input per input neuron.
            inp_maps (:obj:`list` of int): Has one time-to-first-spike per input.
            out_maps (:obj:`list` of int): Has one time-to-first-spike per output.

        Returns:
            Output of the network for the particular `problem` and `inputs`.

        """
        self.reset()
        if problem == 'XOR':
            return self.classify_xor(inputs, inp_maps, out_maps)
        self.reset()


    def classify_xor(self, inputs, inp_maps, out_maps):
        """Solve the XOR problem.

        Input1  Input2  Output
        0       0       0
        0       1       1
        1       0       1
        1       1       0

        Args:
            inputs (:obj:`list` of :obj:`int`): Two values, both `0` or `1`.
            inp_maps (:obj:`list` of :obj:`int`): Two values, input ttfs for 0 and 1.
            out_maps (:obj:`list` of :obj:`int`): Two values, output ttfs for 0 and 1.

        Returns:
            Currently, the time-to-first-spike of the output neuron.

        """
        t_window = 50
        inp_to_curr = {i : self.calibrate_current(inp_maps[i])
                       for i in range(len(inputs))}

        for c_time in range(1, 1000):

            if self.layers[-1][0].last_spike != -1:
                return self.layers[-1][0].last_spike

            for i_c_layer in range(len(self.layers)):
                for i_c_neuron in range(len(self.layers[i_c_layer])):
                    neuron = self.layers[i_c_layer][i_c_neuron]
                    if i_c_layer == 0: curr = inp_to_curr[inputs[i_c_neuron]]
                    else: curr = self.sum_epsps(c_time, i_c_layer, i_c_neuron)
                    neuron.time_step(curr, c_time)
                    #if i_c_layer == 2 and neuron.last_spike == c_time: print('\toutput spikes!')
                    #elif neuron.last_spike == c_time: print('\ttime {}: layer {} neuron {} type {} spikes!'.format(c_time, i_c_layer, i_c_neuron, neuron.type))
                    if VERBOSE:
                        print('t: {}, l: {}, n: {}, sp: {}, c: {}, v: {}'
                            .format(c_time, i_c_layer, i_c_neuron,
                                    neuron.last_spike, curr, neuron.voltage))

            if VERBOSE: print('')


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
        const = Neuron().v_max
        c_layer = self.layers[i_c_layer]
        p_layer = self.layers[i_c_layer-1]
        c_synapses = self.synapses[i_c_layer-1]
        for i_p_neuron in range(len(p_layer)):
            p_neuron = p_layer[i_p_neuron]
            p_spike = p_neuron.last_spike
            if p_spike != -1: # ttfs exists
                weight = c_synapses[i_p_neuron][i_c_neuron]
                change = weight * const * math.e**(-(c_time-p_spike)/self.time_const)
                if VERBOSE: print('\tadding epsp to current: {}'.format(change))
                current += change
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
                return current/10.0


    def reset(self):
        """Reset time-to-first-spikes for every neuron in the network."""
        for layer in self.layers:
            for neuron in layer:
                neuron.last_spike = -1


if __name__ == '__main__':
    snn = SNN([2, 10, 1])
    problems = defaultdict()
    #problems['XOR'] = [[0, 1], [2, 5], [14, 10]]
    problems['XOR'] = [[0, 1], [10, 15], [5, 8]]

    if VERBOSE:
        print('Neuron values:')
        pp(vars(Neuron()))
        print('')

        for ttfs in range(1, 8, 1):
            print('ttfs: {}, current: {}'.format(
                ttfs, snn.calibrate_current(ttfs)))

    """ temporal
    for name, vals in problems.items():
        inps, inp_maps, out_maps = vals
        print('\n{}\n{}'.format(name, '-'*len(name)))
        snn.train(name, inp_maps, out_maps)
        for x in range(1000):
            #print('Training...')
            snn.train(name, inp_maps, out_maps)
            #print('Classifying...')
            expect_zero, expect_one = [], []
            for inp in list(product(inps, inps)): # all input pairs
                if inp[0] == inp[1]: expect_zero.append(snn.classify(name, inp, inp_maps, out_maps))
                else: expect_one.append(snn.classify(name, inp, inp_maps, out_maps))
            expect_zero = list(filter(lambda x: x != None, expect_zero))
            expect_one = list(filter(lambda x: x != None, expect_one))
            if len(expect_zero) == 2 and len(expect_one) == 2:
                diff_zeros = abs(expect_zero[0]-expect_zero[1])
                diff_ones = abs(expect_one[0]-expect_one[1])
                diff_bw = 1000
                for x in expect_zero:
                    for y in expect_one:
                        if diff_bw > abs(x-y):
                            diff_bw = abs(x-y)
                if diff_bw < diff_zeros or diff_bw < diff_ones:
                    print('FAILURE (0,0): {}\t(0,1): {}\t(1,0): {}\t(1,1): {}'.format(expect_zero[0], expect_one[0], expect_one[1], expect_zero[1]))
                else:
                    print('SUCCESS (0,0): {}\t(0,1): {}\t(1,0): {}\t(1,1): {}'.format(expect_zero[0], expect_one[0], expect_one[1], expect_zero[1]))
    """
