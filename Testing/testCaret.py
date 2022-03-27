from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error

# We are going to do the evaluation in the test dataset. We are going to use Pearson Correlation and
# MSE as metrics, with p-value as an indicator for the correlation.

dataset = load_dataset( 'csv', data_files='QATest.csv', split='train')
questions = [str(i) for i in dataset['Question']]
answers = [str(i) for i in dataset['Answer']]
scores = [float(i) for i in dataset['Score']]

model = SentenceTransformer('sbertCaretAmp/')
embeddings1 = model.encode(questions, convert_to_tensor=True)
embeddings2 = model.encode(answers, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1,embeddings2)
c_scores = []
for i in range(len(questions)):
        c_scores.append(cosine_scores[i][i].cpu().detach().numpy())
cor, p = pearsonr(np.array(scores), np.array(c_scores))
mse = mean_squared_error(np.array(scores),np.array(c_scores))
print("Correlation is "+ str(cor) +" and p-value is " + str(p)+". MSE = " + str(mse))
