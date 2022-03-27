from pathlib import Path
import pandas as pd
from haystack.nodes import QuestionGenerator, FARMReader
from haystack.pipelines import QuestionAnswerGenerationPipeline
from haystack.document_stores import InMemoryDocumentStore
import random

#
# In this script we are building the dataset. We are using a .csv from the caret website that
# gives all the details of all the models.
# 

# We are using lists of strings to build our dataset with the Pandas library.

data = []
texts = []
qa = []

tag_data = pd.read_csv("tag_data.csv")
names = tag_data.columns
models = tag_data[names[0]]
names = names[1:]

# Iterating through the information, we are building the descriptions for the models and some questions for each characteristic
for index,row in tag_data.iterrows():
    sentence = str(models[index]) + "is a machine learning model."
    for x in names:
        if row[x] == 1:
            if x == "Classification" or x == "Regression":
                sentence += ("It is used for " + x + ". ")
                qa.append([("Model for " + x + "?"), str(models[index]), (0.9 + random.uniform(-0.1,0.1))])
            else:
                if "Model" in x:
                    sentence += "It is a " + x + ". "
                    qa.append([(x + "?"), str(models[index]), (0.7 + random.uniform(-0.1,0.1))])
                elif "Accepts" in x:
                    sentence += "It " + x + ". "
                    qa.append([("What model " + x + "?"), str(models[index]), (0.8 + random.uniform(-0.1,0.1))])
                else:
                    sentence += "It has " + x + ". "
                    qa.append([("Model that has " +x + "?"), str(models[index]), (0.85 + random.uniform(-0.1,0.1))])
        elif random.random() >= 0.8: # Randomly gives a negative answer
            if x == "Classification" or x == "Regression":
                sentence += ("It is used for " + x + ". ")
                qa.append([("Model for " + x + "?"), str(models[index]), (-0.9 + random.uniform(-0.1,0.1))])
            else:
                if "Model" in x:
                    sentence += "It is a " + x + ". "
                    qa.append([(x + "?"), str(models[index]), (-0.7 + random.uniform(-0.1,0.1))])
                elif "Accepts" in x:
                    sentence += "It " + x + ". "
                    qa.append([("What model " + x + "?"), str(models[index]), (-0.8 + random.uniform(-0.1,0.1))])
                else:
                    sentence += "It has " + x + ". "
                    qa.append([("Model that has " +x + "?"), str(models[index]), (-0.85 + random.uniform(-0.1,0.1))])
    data.append([models[index], sentence])
    texts.append(sentence)

# We build a Pandas dataframe and then export it to csv.
df1 = pd.DataFrame(data, columns=['Title','Descrption'])
df1.to_csv(str(Path(__file__).parent) + '/Descriptions.csv', index=False, header=True)

# Now, we are using Haystack to build some more questions. This library has a pipeline
# that builds questions on a given text and then tries to answer it, with a given score. 
# We are using the RoBERTa base squad2 model to make the answers (is the best available model in this library).

model = "deepset/roberta-base-squad2"
reader = FARMReader(model, use_gpu=True)
qg = QuestionGenerator()
document_store = InMemoryDocumentStore()

qag_pipeline = QuestionAnswerGenerationPipeline(qg, reader)
docs = [ {"content": i} for i in texts]
document_store.write_documents(docs)

for idx, document in enumerate(document_store):
    result = qag_pipeline.run(documents=[document])
    for x in result["results"]:
        if x["answers"]:
            qa.append([x["query"],x["answers"][0].answer,x["answers"][0].score])

# Finally, we create the second dataframe and export it.
df2 = pd.DataFrame(qa, columns=['Question','Answer','Score'])
df2.to_csv(str(Path(__file__).parent) + '/QADataset.csv', index=False, header=True)