#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ['Farrah', 'Fred', 'Felicia']
barWidth = 0.5
fig, ax = plt.subplots()
for i in range(len(fruit)):
    ax.bar(labels, fruit[i], barWidth, label='apples' if i == 0 else 'bananas' if i == 1 else 'oranges' if i == 2 else 'peaches',color='red' if i == 0 else 'yellow' if i == 1 else '#ff8000' if i == 2 else '#ffe5b4', bottom=np.sum(fruit[:i], axis=0))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()