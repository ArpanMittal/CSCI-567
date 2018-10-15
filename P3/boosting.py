import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs  # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples

        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        ########################################################
        # TODO: implement "predict"
        ########################################################
        result = np.zeros(len(features))
        for i in range(self.T):
            result += self.betas[i] * np.array(self.clfs_picked[i].predict(features))
        # print(result)
        temp = [-1 if row <= 0 else 1 for row in result]

        # return temp
        return temp
    def train(self, features: List[List[float]], labels: List[int]):
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples



        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''
        ############################################################
        # TODO: implement "train"
        N = len(labels)
        D = np.ones(N) / N

        for t in range(self.T):
            sum = list()
            for clf in self.clfs:
                h = clf.predict(features)

                sum_2 = 0
                for temp in range(len(h)):

                    if (labels[temp] != h[temp]):
                        sum_2 += D[temp]
                sum.append(sum_2)
            ht = np.argmin(sum)
            self.clfs_picked.append(list(self.clfs)[ht])

            epsilon = sum[np.argmin(sum)]

            if (epsilon != 0):
                beta = 1 / 2 * np.log((1 - epsilon) / epsilon)
            h_final = list(self.clfs)[ht].predict(features)



            self.betas.append(beta)
            for i in range(N):
                if labels[i] == h_final[i]:
                    D[i] *= np.exp(-beta)
                else:
                    D[i] *= np.exp(beta)

            D /= np.sum(D)



    ############################################################

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return






    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)



