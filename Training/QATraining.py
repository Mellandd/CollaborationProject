from transformers import BertForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import numpy as np
from torch.utils.data import DataLoader
import torch

# We are going to train our model with the training dataset.
# We use the sentence_transformers library.

# First we load the dataset and preprocess it for our model.

torch.cuda.empty_cache()
dataset = load_dataset( 'csv', data_files='QATraining.csv', split='train')
questions = [str(i) for i in dataset['Question']]
answers = [str(i) for i in dataset['Answer']]
scores = [float(i) for i in dataset['Score']]
pairs = list(zip(questions, answers,scores))

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# This is the information we are giving to the model. We need our data (with the annotated score)
# and a loss function.

training_data = [InputExample(texts= [a,b], label= c) for (a,b,c) in pairs]
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# We use the evaluator to see the evolution of our metrics in each epoch.
evaluator = evaluation.EmbeddingSimilarityEvaluator(questions, answers, scores)

# Then, we train and save it.
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=15, warmup_steps=10, 
evaluator=evaluator, evaluation_steps=10, scheduler="warmupcosine", output_path= "sbertCaretAmp")
