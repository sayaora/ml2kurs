import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Определим набор данных
data = np.array([
    [-2, -1, 0],  # Алиса
    [25, 6, 1],  # Боб
    [17, 4, 8],  # Чарли
    [-15, -6, -3],  # Диана
    [-12, -7, -3],
    [26, 7, 3]
], dtype=np.float32)

all_y_trues = np.array([
    1,  # Алиса
    0,  # Боб
    0,  # Чарли
    1,  # Диана
    1,
    0
], dtype=np.float32)

# Конвертация данных в тензоры
X = torch.tensor(data)
y = torch.tensor(all_y_trues).view(-1, 1)  # Изменяем форму для соответствия выходу


# Определение модели
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # Скрытый слой с 2 нейронами
        self.fc2 = nn.Linear(2, 1)  # Выходной слой с 1 нейроном
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Создание модели
model = SimpleNN()

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
epochs = 600
for epoch in range(epochs):
    model.train()  # Устанавливаем модель в режим обучения
    optimizer.zero_grad()  # Обнуляем градиенты

    # Прямой проход
    output = model(X)

    # Расчёт потерь
    loss = criterion(output, y)

    # Обратный проход и оптимизация
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Прогноз
masha = torch.tensor([-7, -3, 2], dtype=torch.float32).view(1, -1)  # Преобразуем в нужный формат
vanya = torch.tensor([20, 1, 2], dtype=torch.float32).view(1, -1)
charlie = torch.tensor([19, 1, 3], dtype=torch.float32).view(1, -1)

print("Маша: %.3f" % model(masha).item())  # Предсказание для Маши
print("Ваня: %.3f" % model(vanya).item())  # Предсказание для Вани
print("Чарли: %.3f" % model(charlie).item())  # Предсказание для Чарли

# Сохранение модели
torch.save(model.state_dict(), 'classification_model.pth')