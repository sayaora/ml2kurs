# http://localhost:5000/api?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2

from requests import get
sepal_length = input('Введите sepal_length = ')
sepal_width = input('Введите sepal_width = ')
petal_length = input('Введите petal_length = ')
petal_width = input('Введите petal_width = ')
print(get(f'http://localhost:5000/api?sepal_length={sepal_length}&sepal_width={sepal_length}&petal_length={sepal_length}&petal_width={sepal_length}').json())