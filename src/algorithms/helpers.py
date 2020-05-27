import numpy as np
import pandas as pd


def average_sequences(sequences, sequence_length, output_shape, stride=1):
    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    sequences = np.concatenate(sequences)
    lattice = np.full((sequence_length, *output_shape), np.nan) 
    for i, sequence in enumerate(sequences):
        lattice[i % sequence_length, i:i + sequence_length] = sequence
    sequences = np.nanmean(lattice, axis=0).T
    return sequences


def make_sequences(data, sequence_length, stride=0):
    if not isinstance(data, (pd.DataFrame, list)):
        raise TypeError('data must be of type pd.DataFrame or list.'
            'The type of data was {}.'.format(type(data)))

    if isinstance(data, pd.DataFrame):
        data = [data]

    seq_list = []
    for df in data:
        df = df.interpolate()
        df = df.bfill()
        df = df.values

        n_sequences = df.shape[0] - sequence_length + 1
        sequences = [df[i:i + sequence_length] for i in range(n_sequences)]
        sequences = np.array(sequences)
        seq_list.append(sequences)

    sequences = np.vstack(seq_list)
    return sequences


def split_sequences(sequences, percentage):
    if percentage > 1 or percentage < 0:
        raise ValueError('percentage must be within [0,1],'
                'was {}'.format(percentage))
    indices = np.random.permutation(len(sequences))
    split_point = int(len(sequences) * percentage)
    seq1 = sequences[indices[:split_point]]
    seq2 = sequences[indices[split_point:]]
    return seq1, seq2
