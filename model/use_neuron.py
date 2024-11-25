import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data = np.array([[9, 9]])
predictions = new_neuron.forward(test_data)
print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Помидор', 'Огурец'))