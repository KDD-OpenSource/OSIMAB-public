from pandas import DataFrame
import numpy as np


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
    if not isinstance(data, (DataFrame, np.array)):
        raise TypeError('data must be of type pd.DataFrame or np.array.'
            'The type of data was {}.'.format(type(data)))

    if len(data.shape) not in [2, 3]:
        raise ValueError('sequence: data has wrong numer of dimensions.'
            'requres 2 or 3, has{}.'.format(len(data.shape)))

    if isinstance(data, DataFrame):
        data = data.interpolate()
        data = data.bfill()
        data = data.values

    if len(data.shape) == 3:
        return data

    n_sequences = data.shape[0] - sequence_length + 1
    sequences = [data[i:i + sequence_length] for i in range(n_sequences)]
    sequences = np.array(sequences)
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
