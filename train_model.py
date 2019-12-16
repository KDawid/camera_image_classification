from dataset_collector import DatasetCollector
import lenet


class ModelTrainer:
    def __init__(self, training_set_folder):
        self.collector = DatasetCollector(training_set_folder)
        self.label_dict = self.collector.get_label_dict()
        self.labels_num = self.collector.get_labels_num()
        self.model = lenet.get_model(self.labels_num, (50, 50, 3))
        self.history = None

    def train(self, epochs=1):
        x_train, y_train = self.collector.get_data((50, 50))
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=128)

    def load_weights(self, file_path='model.h5'):
        self.model.load_weights(file_path)

    def save_weights(self, file_path='model.h5'):
        self.model.save_weights(file_path)

    def get_model(self):
        return self.model

    def get_label_dict(self):
        return self.label_dict
