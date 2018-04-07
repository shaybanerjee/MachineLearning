import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Svm:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if (self.visualization):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        self.data = data
        opt_dict = {}
        transforms = [[1,1], [-1, 1], [-1. -1], [1, -1]]

        all_data = []
        for yi in self.data:
            for features in self.data[yi]:
                for feature in features:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        b_range_multiple = 5
        # No need to take as small steps for b unlike w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # Convex problem, becomes True when we have Global min
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, b_multiple):
                    for transformation in transforms:
                        w_transformed = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                if not i*(np.dot(w_transformed,xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_transformed)] = [w_transformed, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w  = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features, self.w + self.b)))
        return classification


data_dict = {-1: np.array([ [1,7], [2,8], [3,8], ]), 1: np.array([  [5,1], [6,-1], [7,3],  ])}




