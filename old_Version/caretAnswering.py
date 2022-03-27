from transformers import BertForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import numpy as np
from torch.utils.data import DataLoader


dataset = load_dataset( 'csv', data_files='caretQuestions.csv', split='train')
questions = [str(i) for i in dataset['Question']]
answers = [str(i) for i in dataset['Answer']]
pairs = list(zip(questions, answers))
scores = [0.9 for i in range(0,len(questions))]
noise = np.random.normal(0,.1, len(scores)).tolist()
scores = [a+b for a,b in zip(scores,noise)]


model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
training_data = [InputExample(texts= [a,b], label= 1.0) for (a,b) in pairs]
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

evaluator = evaluation.EmbeddingSimilarityEvaluator(questions, answers, scores)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100, 
evaluator=evaluator, evaluation_steps=100, output_path= "sbertCaret")