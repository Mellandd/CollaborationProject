import pandas as pd
from pathlib import Path
import nlpaug.augmenter.char as nac
from datasets import load_dataset
import nlpaug.flow as naf
import nlpaug.augmenter.word as naw

training2 = []
training3 = []

dataset = load_dataset( 'csv', data_files='caretTraining.csv', split='train')
questions = [str(i) for i in dataset['Question']]
answers = [str(i) for i in dataset['Answer']]
scores = [float(i) for i in dataset['Score']]

aug = naf.Sequential([
    nac.RandomCharAug(action="substitute"),
    naw.RandomWordAug()
])

aug2 = naw.WordEmbsAug(
        model_type='glove',model_path='glove.6B.100d.txt',
        action="substitute")


for i in range(len(questions)):
    training2.append([questions[i],answers[i],scores[i]])
    training3.append([questions[i],answers[i],scores[i]])
    q = aug.augment(questions[i])
    a = aug.augment(answers[i])
    s = scores[i] * 0.85
    training2.append([q,a,s])
    q = aug.augment(questions[i])
    a = aug.augment(answers[i])
    training2.append([q,a,s])

    q = aug2.augment(questions[i])
    a = aug2.augment(answers[i])
    s = scores[i] * 0.9
    training3.append([q,a,s])
    q = aug2.augment(questions[i])
    a = aug2.augment(answers[i])
    training3.append([q,a,s])

df2 = pd.DataFrame(training2, columns =['Question', 'Answer', 'Score'])

df2.to_csv(str(Path(__file__).parent) + '/caretTraining2.csv', index=False, header=True)


df3 = pd.DataFrame(training3, columns =['Question', 'Answer', 'Score'])

df3.to_csv(str(Path(__file__).parent) + '/caretTraining3.csv', index=False, header=True)

