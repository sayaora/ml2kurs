import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[5, 6],
              [7, 7],
              [3, 9],
              [4, 10],
              [5, 5],
              [3, 8],
              [5, 10]])
y = np.array([1, 1, 0, 0, 1, 0, 0])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=2)
neuron.train(X, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')