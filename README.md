# 📰 News Classification

Using DistilBERT output to train a classifier in Vowpal Wabbit, in order to sort news articles into one of 24 categories, and compare performance to a classifier trained directly on vectorized data.

## <img src="https://i.imgur.com/S7Um8RF.png" width="24px"> Data

I used a Kaggle dataset found [here]('https://www.kaggle.com/datasets/rmisra/news-category-dataset'). It contains information about nearly 210,000 HuffPost news headlines from 2012 to 2022. Each record in the dataset consists of the following attributes: category, headline, authors, link, short_description and date. There are 26 unique categories of news articles represented in the first 2500 instances, which will be predicted based on the article's short description, headline and BERT output.

## <img src="https://i.imgur.com/pjiwkDp.png" width="24px"> Modeling

DistilBERT takes as input a sequence of words, and passes them through a stack of encoders which output a sequence of vectors corresponding to each token of the input sequence, including special tokens such as CLS and SEP. The CLS token is added to the start of each input sequence, and contains high-level information about the sequence as a whole. The output vector corresponding to the CLS token can be used for classification, and allows the model to make predictions based on the overall meaning of the input, rather than just local context.

I used a pre-defined tokenizer and a pre-trained model from DistilBERT's base, imported from HuggingFace, to extract CLS tokens from training data. I trained a classifier with a logisic loss function in Vowpal Wabbit. I then trained the same type of classifier on the original data, but without the CLS token information.

## 🔍 Further research
* Experiment with different transformer-based models, such as RoBERTa or GPT-3
* Investigate the impact of using larger datasets on model performance, especially in Vowpal Wabbit
* Explore the use of other classifiers, such as Neural Networks, and see if they yield better performance
* Experiment with hyperparameter tuning
* Experiment with different input data types
* Investigate the potential for transfer learning by training the model on one dataset and fine-tuning it on a related dataset

