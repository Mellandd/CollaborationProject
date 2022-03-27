import streamlit as st
import sentence_transformers
from transformers import BertForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
import torch

@st.cache()
def prediction(question):
    dataset = load_dataset( 'csv', data_files='caretDescriptions.csv', split='train')
    descriptions = list(dataset['Description'])
    model = SentenceTransformer('sbertCaret/')
    embeddings = model.encode(descriptions, convert_to_tensor=True)

    question_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, question_embedding)
    a = torch.argmax(cosine_scores)
    return dataset['Title'][a]

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Caret Q&A ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    question = st.text_input("Put your question here")
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Answer"): 
        result = prediction(question) 
        st.success(result)


if __name__=='__main__': 
    main()