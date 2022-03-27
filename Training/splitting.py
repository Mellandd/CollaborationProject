import pandas as pd
from pathlib import Path

#This script splits the dataset into training and test.

df = pd.read_csv('/Users/melland/Universidad/CollaborationProject/caret/QADataset.csv')
train = df.sample(frac=0.9,random_state=200)
test = df.drop(train.index)
train.to_csv(str(Path(__file__).parent) + '/caretTraining.csv', index=False, header=True)
test.to_csv(str(Path(__file__).parent) + '/caretTest.csv', index=False, header=True)

