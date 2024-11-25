# Вызов только из приложения
from requests import get
sepal_length = input('Введите sepal_length = ')
sepal_width = input('Введите sepal_width = ')
petal_length = input('Введите petal_length = ')
petal_width = input('Введите petal_width = ')
print(get('http://localhost:5000/api_v2', json={'sepal_length':sepal_length,'sepal_width':sepal_width, 'petal_length':petal_length, 'petal_width':petal_width}).json())

# {
#     "sepal_length":1,
#     "sepal_width":2,
#     "petal_length":3,
#     "petal_width":4
# }