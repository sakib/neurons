
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

    """ Function to obtain in-order tuples on a list.

    Args:
        iterable (:obj:`list`): Any iterable on a collection. For example:
            [s0, s1, s2, s3, ...]

    Returns:
        :obj:`list` of :obj:`tuple`: Pairwise tuples from input collection:
            [(s0, s1), (s1, s2), (s2, s3), ...]

    """

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

        """Initialize an SNN with random weights in range [0, 1].

        Args:
            n_layers (:obj:`list` of :obj:`int`): Each entry is a number which
                corresponds to the number of neurons in that layer.
        """

        """For a specific problem, train the SNN.

        Args:
            problem (str): Specifies the problem name. For example: 'XOR'
            inp_maps (:obj:`list` of int): Has one time-to-first-spike per input.
            out_maps (:obj:`list` of int): Has one time-to-first-spike per output.

        """

        """Train the SNN to solve the XOR problem.

        Args:
            inp_maps (:obj:`list` of `int`): Two values, input ttfs for 0 and 1.
            out_maps (:obj:`list` of `int`): Two values, output ttfs for 0 and 1.

        """

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

        """Update weights of all synapses in the SNN.

        Args:
            c_time (int): Current timestep in the SNN.
            t_window (int): Length of the time window in the SNN to check for spikes in.

        """

        """For a specific problem, run the SNN until output neuron fires.

        Args:
            problem (str): Specifies the problem name. For example: 'XOR'
            inputs (:obj:`list` of :obj:`int`): Has one input per input neuron.
            inp_maps (:obj:`list` of int): Has one time-to-first-spike per input.
            out_maps (:obj:`list` of int): Has one time-to-first-spike per output.

        Returns:
            Output of the network for the particular `problem` and `inputs`.

        """

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

        """Get the electric current constant needed to make neuron spike first.

        Args:
            ttfs (int): Target time-to-first-spike for the neuron.

        Returns:
            Value of constant current for which, if you supplied it to the
            default neuron, would make it spike first at time `ttfs`.

        """
