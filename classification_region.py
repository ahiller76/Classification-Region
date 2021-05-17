### Aaron Hiller, Kit Sloan
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

input = np.array([[1, 5], [2, 4], [7, 7], [4, 6], [6, 4], [6, 9], [4, 2], [8, 6], [5, 5], [3, 8]], "float32")
colors = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])

#Normalizing input data X and Y separatly
normal_input = np.copy(input)
for i in range(0, 2):
   normal_input[:, i] = (input[:, i] - np.min(input[:, i])) / (np.max(input[:, i]) - np.min(input[:, i]))

# 2 inputs, 3 hidden layers with 60 nodes , and a softmax output.
model = models.Sequential()
model.add(layers.Dense(60, activation='sigmoid', input_shape=(2,)))
model.add(layers.Dense(60, activation='sigmoid'))
model.add(layers.Dense(60, activation='sigmoid'))
model.add(layers.Dense(2, activation='softmax'))
model.compile( optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(normal_input, colors, epochs=7500)
test_loss, test_acc = model.evaluate(normal_input, colors)
print('Test Accuracy:', test_acc)

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

step = 0.1
interval = int(10 / step)
size = interval * interval
step_div = int(interval / 10)

predict_input = np.ndarray(shape=(size, 2))
for i in range(0, int(interval)):
    for j in range(0, int(interval)):
        predict_input[j + i * interval, 0] = i/step_div
        predict_input[j + i * interval, 1] = j/step_div

# Normalize predicted input
for i in range(0,2):
    predict_input[:, i] = (predict_input[:, i] - np.min(input[:, i])) / (np.max(input[:, i]) - np.min(input[:, i]))

plt.figure(1)
classes = model.predict(predict_input)

for i in range(0, interval):
    for j in range(0, interval):
        if classes[j + (i * interval), 0] > classes[j + (i * interval), 1]:
            plt.plot(i/step_div, j/step_div, 'bo', markersize=1)
        else:
            plt.plot(i/step_div, j/step_div, 'ro', markersize=1)

    print("Plotting:" , int((i / interval) * 100) , "%")

# Plotting for classification
plt.scatter(input[0:5, 0], input[0:5, 1], c="r")
plt.scatter(input[6:, 0], input[6:, 1], c="b")
axes = plt.gca()
axes.set_ylabel('y')
axes.set_xlabel('x')
axes.set_title('Classification Area')
axes.set_xlim([0, 10])
axes.set_ylim([0, 10])
plt.show()

# Plotting the loss over time
plt.figure(2)
plt.plot(epochs, loss, label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()