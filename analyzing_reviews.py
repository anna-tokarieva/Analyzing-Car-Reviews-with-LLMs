# Import necessary packages
import pandas as pd
import torch

from transformers import logging
logging.set_verbosity(logging.WARNING)

#TASK 1: Use a pre-trained LLM to classify the sentiment of the five car reviews in the car_reviews.csv dataset, and evaluate the classification accuracy and F1 score of predictions.
#Step 1: Loading the CSV
path = 'data/car_reviews.csv'
file = pd.read_csv(path, delimiter = ';')
#file.head()

#Step 2: extracting car comments and evaluation
review_label = file['Class'].tolist()
review_text = file['Review'].tolist()
#print(review_label)
#print(review_text)

#Step 3: loading sentiment analysis
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model = 'distilbert-base-uncased-finetuned-sst-2-english')

#Step 4: running it on review text
predicted_labels = sentiment(review_text)

#Step 5: Converting model labels into 0/1
predictions = []
for item in predicted_labels:
    if item['label'] == 'POSITIVE':
        predictions.append(1)
    else: 
        predictions.append(0)
#print(predictions)

#step 6: converting real labels into 0/1
real_labels = []
for item in review_text:
    if item == 'POSITIVE':
        real_labels.append(1)
    else:
        real_labels.append(0)
#print(real_labels)

#Step 7: accuracy and F1
from sklearn.metrics import accuracy_score, f1_score
accuracy_result =  accuracy_score(predictions, real_labels)
f1_result = f1_score(predictions, real_labels)
print('Accuracy:', accuracy_result)
print('F1 result:', f1_result)

#TASK 2: The company is recently attracting customers from Spain. Extract and pass the first two sentences of the first review in the dataset to an English-to-Spanish translation LLM. Calculate the BLEU score to assess translation quality, using the content in reference_translations.txt as references. 
#Step 1: translating text
first_review = review_text[0]
sentences_split = first_review.split('.')
#print(split)
two_sentences = sentences_split[0] + '. ' + sentences_split[1] + '.'
#print(two_sentences)
translation = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translation(two_sentences)[0]['translation_text']
print(translated_review)

#Step 2: calculating bleu score
import evaluate
bleu = evaluate.load('bleu')
reference_text = open('data/reference_translations.txt').read()
bleu_score = bleu.compute(predictions=[translated_review], references = [[reference_text]])
print(bleu_score)

#Task 3: The 2nd review in the dataset emphasizes brand aspects. Load an extractive QA LLM such as "deepset/minilm-uncased-squad2" to formulate the question "What did he like about the brand?" and obtain an answer.
from transformers import pipeline
question_answer = pipeline('question-answering', model = 'deepset/minilm-uncased-squad2')

context = review_text[1]
#print(context)
question = 'What did he like about the brand?'
question_answer_output = question_answer(question = question, context = context)
answer = question_answer_output['answer']
print('Answer: ', answer)

#Task 4: Summarize the last review in the dataset, into approximately 50-55 tokens long. Store it in the variable summarized_text.
last_review = review_text[-1]
print(last_review)

from transformers import pipeline
summarization = pipeline('summarization', model = 'cnicu/t5-small-booksum')
summary = summarization(last_review, max_length = 53)
summarized_text = summary[0]['summary_text']
print(f'Summarized text:\n{summarized_text}')

