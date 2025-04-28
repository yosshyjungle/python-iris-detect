import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
iris = datasets.load_iris()

from sklearn.model_selection import train_test_split as split
x_train, x_test, y_train, y_test = split(iris.data, iris.target, test_size=0.2)

model = keras.models.Sequential([
    keras.layers.Flatten(input_dim=4),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer='sgd',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

score = model.evaluate(x_test, y_test)
print('loss:', score[0])
print('accuracy:', score[1])
model.save('iris_model.keras')
#一つのデータに対する評価の実行
# import numpy as np
# x = np.array([[5.1, 3.5, 1.4, 0.2]])
# r = model.predict(x)
# print(r)
# r.argmax()


