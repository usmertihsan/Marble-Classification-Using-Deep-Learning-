from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels = ['6 Class','7 class with otherType', '25 Class']
Positive = [97,         87,    91]
cmap = plt.cm.tab10
colors = cmap(np.arange(len(labels)) % cmap.N)

x = np.arange(len(labels))
width = 0.6  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/30, Positive, width, color=colors)

ax.set_ylabel('Accuracy')
plt.xticks(x, labels, rotation='vertical')
ax.set_xticklabels(labels)
ax.legend()
plt.show()