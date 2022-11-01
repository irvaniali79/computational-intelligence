from utils.Cilib import *
import matplotlib.pyplot as plt
import numpy as np
inputdata = [
    file_into_numpy("./coordinators/1.txt"),
    file_into_numpy("./coordinators/2.txt"),
    file_into_numpy("./coordinators/3.txt"),
    file_into_numpy("./coordinators/4.txt"),
    file_into_numpy("./coordinators/5.txt"),
    file_into_numpy("./coordinators/6.txt")
]
for index in range(len(inputdata)):
    net_input = inputdata[index]
    np.random.shuffle(net_input)
    x_train, y_train = x_y_split(net_input)

    adaline = Adaline()

    coefs, epoc = adaline.fit(x_train, y_train)
    print(epoc)
    plotNodes(x_train, y_train)
    plotLine(coefs)

    y_hat = adaline.classify(x_train)
    print(adaline.score(y_train, y_hat))

    plt.axis()
    plt.show()
