import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


class ConfusionMatrix:
    def __init__(self, predictions, actual):
        self.header = self._create_header(predictions, actual)
        self.matrix = self._create_matrix(predictions, actual)

    @staticmethod
    def _create_header(predictions, actual):
        result = set()
        for p in predictions:
            result.add(p)
        for a in actual:
            result.add(a)
        return sorted(list(result))

    def _create_matrix(self, predictions, actual):
        result = [[0 for i in self.header] for j in self.header]

        data = []
        for i in range(len(predictions)):
            item = (predictions[i], actual[i])
            data.append(item)

        for (pred, act) in data:
            predicted_index = self.header.index(pred)
            actual_index = self.header.index(act)
            result[actual_index][predicted_index] += 1
        return result

    def get_hit_rate(self, as_string=True):
        correct = sum(self.matrix[i][i] for i in range(len(self.matrix)))
        hit_rate = correct/np.sum(self.matrix)
        if as_string:
            return "{0:.0f}%".format(hit_rate*100)
        return hit_rate

    @staticmethod
    def extendMatrix(matrix):
        result = matrix
        result.append([])
        for i in range(len(matrix) - 1):
            result[i].append(0)
            result[len(result) - 1].append(0)
        result[len(result) - 1].append(0)
        return result

    def getRowStatistic(self, matrix, i):
        correct = matrix[i][i]
        all_data = np.sum(matrix[i])
        value = correct / all_data
        if value != value:
            return "-"
        return self.get_accuracy_str(value)

    def getColumnStatistic(self, matrix, i):
        correct = matrix[i][i]
        all_data = np.sum([matrix[n][i] for n in range(len(matrix))])
        value = correct / all_data
        if value != value:
            return "-"
        return self.get_accuracy_str(value)

    @staticmethod
    def get_accuracy_str(value):
        return "{0:.0%}".format(value)

    def extendMatrixWithStatistics(self, np_matrix):
        result = [""] * len(np_matrix)
        for i in range(len(result)):
            result[i] = [""] * len(result)
        for i in range(len(result) - 1):
            for j in range(len(result) - 1):
                result[i][j] = str(np_matrix[i][j])
        for i in range(len(result) - 1):
            result[i][len(result) - 1] = self.getRowStatistic(np_matrix, i)
            result[len(result) - 1][i] = self.getColumnStatistic(np_matrix, i)
        result[len(result) - 1][len(result) - 1] = ""

        return np.array(result)

    @staticmethod
    def get_color_values(matrix):
        res = [0] * len(matrix)
        for i in range(len(res)):
            res[i] = [0] * len(matrix)
        for i in range(len(matrix) - 1):
            row_sum = np.sum(matrix[i])
            for j in range(len(matrix[i]) - 1):
                if row_sum > 0:
                    res[i][j] = matrix[i][j] / row_sum
            res[i][len(matrix) - 1] = -0.25
            res[len(matrix) - 1][i] = -0.25
            res[len(matrix) - 1][len(matrix) - 1] = -0.25
        return res

    def create_graph(self, save=None):
        num = np.sum(self.matrix)
        correct = sum([self.matrix[i][i] for i in range(len(self.matrix))])

        graph_matrix = self.extendMatrix(self.matrix)

        graph_axes = self.header
        if len(graph_matrix) != len(graph_axes):
            graph_axes.append("")
        np_count = np.array(graph_matrix)

        df_cm = pd.DataFrame(self.get_color_values(graph_matrix), index=graph_axes, columns=graph_axes)
        plt.figure(figsize=(12, 12))
        heat = sn.heatmap(df_cm, annot=self.extendMatrixWithStatistics(np_count),
                          fmt="s", cmap="Blues", cbar=False, linewidths=0.3)

        title = f"Confusion matrix (#: {num}, hit rate: {self.get_accuracy_str(correct / num)})"
        plt.title(title)

        plt.xlabel("Classified as")
        plt.ylabel("Type")
        plt.yticks(rotation=0)

        if save:
            plt.savefig(save)
        else:
            plt.show()
