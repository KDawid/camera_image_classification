from confusion_matrix import ConfusionMatrix
from dataset_collector import DatasetCollector
import lenet

collector = DatasetCollector()
x_train, y_train = collector.get_data('training', (50, 50))
model = lenet.get_model(len(y_train[0]), (50, 50, 3))

history = model.fit(x_train, y_train, epochs=20, batch_size=128)

predictions = model.predict(x_train)
print([x.index(max(x)) for x in predictions.tolist()])
print([y.index(max(y)) for y in y_train.tolist()])
print()
prediction_list = [collector.label_dict[x.index(max(x))] for x in predictions.tolist()]
actual_list = [collector.label_dict[y.index(max(y))] for y in y_train.tolist()]
print(prediction_list)
print(actual_list)
print()
print(collector.label_dict)

matrix = ConfusionMatrix(prediction_list, actual_list)
matrix.create_graph()
