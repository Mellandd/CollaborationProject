import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(0,10,190)
df = pd.read_csv('sbertCaretAmp/eval/similarity_evaluation_results.csv')

y = np.array(df["euclidean_pearson"][0:190])
y = y.astype(float)

plt.plot(x,y)
plt.ylabel("Correlation")
plt.xlabel("Epochs")
plt.title("Training with data augmentation second version")
plt.show()

