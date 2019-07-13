# twitter-sentiment-analysis
Solution to the practice problem : **[Twitter Sentiment Analysis](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)**

**Problem Statement**
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

## Rsults

* Data preprocess [here](jupyter/1.data_process.ipynb)
* Data Analysis [here](jupyter/2.data_analysis.ipynb)
* Clasical machine leraning aproach [here](jupyter/3.classic_ML.ipynb)
* Neural Networks [here](jupyter/4.neural_networks.ipynb)
* Own keras emmbeding [here](jupyter/5.own_embedding.ipynb)
* GLOVE aproach [here](jupyter/6.GLOVE.ipynb)
* Word2Vec aproach [here](jupyter/7.word2vec.ipynb)
* NumberBatch aproach [here](jupyter/8.NumberBatch.ipynb)
* ELMO aproach [here](jupyter/8.Elmo.ipynb)
* BERT aproach [here](jupyter/9.BERT.ipynb)

**Results in validation set**

| Method             | F1      | ACC     |
|:------------------ |:-------:|:-------:|
|Decision Tree TF-IDF| 47.1195 |         |
|NN with BOW & TF-IDF| 45.2961 |  93.35  |
|Own Embedding       | 57.3957 |  94.04  |
|Own Embedding Deep  |         |         |
|GLOVE               | 52.4426 |  92.53  |
|Word2Vec            | 50.4634 |  92.47  |
|ELMO                | 56.8396 |  94.27  |
|BERT                | 62.4100 |  95.87  |


## Web Implementation
This project was used for train the model for a web app, analys your tweet, go to [here]() to see the repository.
For the demo app click [here](). For the medium post click [here]()

## Built With

* [Keras](https://github.com/keras-team/keras) - Frontend for Deeplearning
* [TensorFlow](https://github.com/tensorflow/tensorflow) - Bakend for Deeplearning
* [Sklearn](http://scikit-learn.org/stable/) - Machine Learning Process
* [Pandas](https://pandas.pydata.org) - Data structures and data 
* [Numpy](http://www.numpy.org/) - Data manipulation
* [Gensim](https://pandas.pydata.org) - Data structures and data 
* [NLTK](https://www.nltk.org/) - Text manipulation and NLP
* [seaborn](https://seaborn.pydata.org/) - Data visualization library based on matplotlib
* [wordcloud](http://amueller.github.io/word_cloud/) - Wordclouds plotting
* [editdistance](https://github.com/aflc/editdistance) - Data structures and data 
* [FasText](https://fasttext.cc/) - Library for efficient text classification
* [Glove](https://nlp.stanford.edu/projects/glove/) - Models for GLOVE (other aproach to word2vec)
* [Word2Vec](https://code.google.com/archive/p/word2vec/)

## Authors

* **Sebastian Correa Echeverri** [scorrea92 gitlab](https://gitlab.com/scorrea92) [scorrea92 github](https://github.com/scorrea92)

## Licence
see LICENCE file