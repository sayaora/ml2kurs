import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torchvision import transforms

from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [
    {"name": "Лаба 1", "url": "p_knn"},
    {"name": "Лаба 2", "url": "p_lab2"},
    {"name": "Лаба 3", "url": "p_lab3"},
    {"name": "Лаба 4", "url": "p_lab4"},
    {"name": "Лаба 5", "url": "p_lab5"}
]

def preprocess_image(file):
    image = Image.open(file.stream).convert('L')  # Преобразуем в градации серого
    image = image.resize((28, 28))  # Изменение размера
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Нормализация
    ])
    image = transform(image).unsqueeze(0)  # Добавляем размерность для батча
    return image

# Загрузка моделей

new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model/neuron_weights.txt')


# Обновленный класс для Keras
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model_class = SimpleNN()
model_class.load_state_dict(torch.load('model/classification_model.pth'))
model_class.eval()

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return self.fc2(F.relu(self.fc1(x)))

model_clothes = FashionMNISTCNN()
model_class.load_state_dict(torch.load('model/classification_model.pth', weights_only=True))
model_clothes.eval()

class_names = [
    'футболка', 'брюки', 'свитер', 'платье',
    'пальто', 'сандали', 'рубашка', 'кеды',
    'сумка', 'ботинки'
]

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы", menu=menu)

@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + str(pred))

@app.route("/p_lab2")
def f_lab2():
    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)

@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)

@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        predictions = new_neuron.forward(X_new)
        result = 'Помидор' if predictions >= 0.5 else 'Огурец'
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + result)

@app.route("/p_lab5", methods=['POST', 'GET'])
def p_lab5():
    if request.method == 'GET':
        return render_template('lab5.html', title="Одежда", menu=menu, class_model='')

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('lab5.html', title="Одежда", menu=menu, class_model="Файл не загружен.")

        file = request.files['image']
        if file.filename == '':
            return render_template('lab5.html', title="Одежда", menu=menu, class_model="Файл пуст.")

        x_new = preprocess_image(file)

        with torch.no_grad():
            predictions = model_clothes(x_new)
            predicted_class_index = torch.argmax(predictions).item()
            predicted_class_name = class_names[predicted_class_index]

        return render_template('lab5.html', title="Одежда", menu=menu, class_model=f"Это: {predicted_class_name}")

@app.route('/api_class2', methods=['GET'])
#GET /api_class2?x1=1.75&x2=70&x3=2.5
def predict_classification2():
    x1 = float(request.args.get('x1'))
    x2 = float(request.args.get('x2'))
    x3 = float(request.args.get('x3'))
    input_data = np.array([[x1, x2, x3]])

    with torch.no_grad():
        predictions = model_class(torch.tensor(input_data, dtype=torch.float32)).item()
    result = 'парень' if predictions >= 0.5 else 'девочка'

    return jsonify(sex=result)

if __name__ == "__main__":
    app.run(debug=True)
