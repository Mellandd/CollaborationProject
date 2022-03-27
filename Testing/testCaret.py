from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats.stats import pearsonr

dataset = load_dataset( 'csv', data_files='caretTraining.csv', split='train')
questions = [str(i) for i in dataset['Question']]
answers = [str(i) for i in dataset['Answer']]
scores = [float(i) for i in dataset['Score']]

model = SentenceTransformer('sbertCaret/')
embeddings1 = model.encode(questions, convert_to_tensor=True)
embeddings2 = model.encode(answers, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1,embeddings2)

cor, p = pearsonr(scores, cosine_scores.cpu().detach().numpy())
print("Correlation is "+ str(cor) +" and p-value is " + str(p))

