import matplotlib.pyplot as plt
from utils.Adaline import *



def x_y_split(data):
    source = []
    target = np.asarray([])

    for record in data:
        source += [record[:-1].tolist()]
        target = np.append(target, record[-1])

    return np.asarray(source), target


def file_into_numpy(address, _type=np.float32):
    with open(address) as f:
        lines = f.readlines()
    return np.asarray([line.split() for line in lines], dtype=_type)


def plotNodes(inputsMatrix, targetVector):
    for record, target in zip(inputsMatrix, targetVector):
        if target == 1:
            plt.plot(*zip(*[record]), marker='o', color='b', ls='')
        else:
            plt.plot(*zip(*[record]), marker='o', color='r', ls='')

def plotLine(coefficients):
    m = -1*coefficients[0]/coefficients[1]
    b3 = -1*coefficients[2]/coefficients[1]
    X = np.linspace(-1, 1)
    Y3 = m*X+b3
    plt.plot(X, Y3, '#008000')

