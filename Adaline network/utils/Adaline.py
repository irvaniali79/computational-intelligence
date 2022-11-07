import math

import numpy as np
from math import inf


class Adaline:
    def __init__(self, epsilon=1, random_state=1, learning_rate=0.5):
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.coef = np.asarray([])
        self.epsilon = epsilon

    def classify(self, x):
        y_hat = np.asarray([])
        column = np.array([[[1]]*len(x)])
        x = np.append(x, column[0], axis=1)

        for index, record in enumerate(x):
            y_in = np.dot(record, self.coef).sum()
            y_hat = np.append(y_hat, 1 if y_in >= 0 else -1)

        return y_hat

    def fit(self, x, t, activition_function=lambda x: x):
        rgen = np.random.RandomState(self.random_state)
        _size = x.shape[1] + 1
        # add biase
        column = np.array([[[1]]*len(x)])
        x = np.append(x, column[0], axis=1)
        self.coef =np.round(np.fabs(rgen.normal(loc=0.1, scale=0.1, size=_size)),4) # + 1 for biase
        largestWight = -1*inf
        firstTime = True
        epoc = 0
        while self.epsilon < largestWight or firstTime:
            firstTime = False
            epoc += 1
            largestWight = -1*inf
            for index, record in enumerate(x):
                X = np.asarray(list(map(activition_function, record)))

                y_in = np.dot(X, self.coef).sum()
                errors = X*self.learning_rate*(t[index] - y_in)
                self.coef += errors
                largest = errors.max()
                if largest > largestWight:
                    largestWight = largest

            print(largestWight)
        return self.coef, epoc

    @staticmethod
    def score(y_target, y_hat):
        trueFalseMapper = {
            1 : True,
            -1 : False,
            0 : False,
        }
        failureCount = 0
        for yTarget, yHat in zip(y_target, y_hat):
            tempYTarget = lambda y: False if y == '-1' or y == -1 else True
            if(trueFalseMapper[yHat] != tempYTarget(yTarget)):
                failureCount += 1
        return (1-(failureCount/y_hat.shape[0]))*100
