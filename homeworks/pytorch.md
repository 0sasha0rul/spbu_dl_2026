# Домашнее задание 1. Введение в Pytorch и DL. Базовые элементы.

## Задание 1.
( 6 баллов) 
  Вам дана базовая нейронная сеть:
  
  class BaseNetwork(nn.Module):
    
    def __init__(self, act_fn, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128]):
        """
        Inputs:
            act_fn - Object of the activation function that should be used as non-linearity in the network.
            input_size - Size of the input images in pixels
            num_classes - Number of classes we want to predict
            hidden_sizes - A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()
        
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        
        self.config = {"act_fn": act_fn.__class__.__name__, "input_size": input_size, "num_classes": num_classes, "hidden_sizes": hidden_sizes} 
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            x = l(x)
        return x
	
	Используйте функции активации  ReLU, tanh, identity. Выберите несколько типов инициализации (константная, случайная, ксавье итп) для каждой постройте графики градиентов и активаций на выходе с каждого слоя. Покажите, какая инициализация приведет к стабильным значениям активаций и градиентов для каждой функции активации. На примере FashionMNIST сравните сходимость модели с различными функциями активации и инициализации для оптимизатора SGD.

	{*} 1 балл. Проведите такой же анализ других функций активации (хотя бы 1). 

	
	## Задание 2. {*}
(5 баллов) Реализуйте базовые функции autograd. Можете вдохновиться видео от Andrej Karpathy. Напишите класс, аналогичный предоставленному классу 'Element', который реализует основные операции autograd: сложение, умножение и активацию ReLU. Класс должен обрабатывать скалярные объекты и правильно вычислять градиенты для этих операций посредством автоматического дифференцирования. Плюсом будет набор предоставленных тестов, оценивающих правильность вычислений. Большим плюсом будет, если тесты будут   написаны с помощью unittest. Можно использовать только чистый torch (без использования autograd и torch.nn). За каждую нереализованную операцию будет вычитаться 3 балла.
	Пример: 
        ```
        a = Node(2)
        b = Node(-3)
        c = Node(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
        Output:  Node(data=2, grad=0)  Node(data=-3, grad=10)  Node(data=10, grad=-3)  Node(data=-28, grad=1)  Node(data=0, grad=1)
        ```
```python

class Node:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op

	def __repr__(self):
		return f" Element(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		 # Implement addition here
		pass

	def __mul__(self, other):
		# Implement multiplication here
		pass

	def relu(self):
		# Implement ReLU here
		pass

	def backward(self):
		# Implement backward pass here
		pass
```
## Задание 3. 
Реализуйте один из оптимизаторов на выбор. Придумайте и напишите тесты для проверки выбранного оптимизатора. 
Также опишите идею алгоритма (+1 балл).
{*} Можете реализовать более 1 алгоритма. Каждый следующий даст 1 балл (максимум 3). 

Варианты:
- Momentum (3 балла)
- Nesterov (3 балла) 
- Adagrad (4 балла)
- Adadelta (4 балла)
- RMSProp (5 баллов)
- Adam (5 баллов)
- Nadam (6 баллов)
- NAG (6 баллов)
- AdamW (6 баллов)

На примере FashionMNIST и модели из классной/д.з покажите сходимость ваших реализаций. Кроме ошибки, выводите значение нормы градиентов и распределения весов.

## Задание 4.
Рассмотрим датасет по предсказанию года создания песни по числовым признакам.
Информацию о датасете можно найти здесь: 
- https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD#
- http://labrosa.ee.columbia.edu/millionsong
- https://en.wikipedia.org/wiki/Timbre

**Ссылка**: https://www.kaggle.com/t/dace4caa40384c86a9879217eaccab42

Мы используем поднабор этого датасета. Пожалуйста, не используйте исходный датасет для читинга, за это получите 0 за ВООБЩЕ всю домашку. 

Попробуйте обучить нейронную сеть (используя линейные и другие слои, возможную нормализацию и активации). Можете использовать любые методы предобработки данных и фокусы, кроме ансамблирования и использования сторонних источников ДАННЫХ. 
- Базовое максимальное число баллов - 6. 
- Топ 25% получат +3 балла, топ 50 +2 балла, топ 75 +1 балл.
- Решение должно быть уникальным, необходимо предоставить код и описание решения. Если использованы внешние источники (туториалы, статьи и т.п.), необходимо предоставить ссылки.
- Обучите модель, используя ваш кастомный оптимизатор. Это даст еще 1 балл. Покажите, изменилось ли что-то.
- При решении задачи можно использовать любые встренные в torch модули (nn, optim, transforms, etc)
- Можно использовать numpy, sklearn, pandas ТОЛЬКО для предобработки и постобработки решения (сборки сабмишена). Для самой модели можно использовать только torch.
- При переобучении можно добавить dropout, batchnorm итд. 
