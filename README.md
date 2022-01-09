Short description of the models:


These models cover different areas of NLP.

- key words:
  -model 1:  Vector space model, term frequency–inverse, classification via k-NN

  -model 2: Logistic Regresjon, Sequence models (NER, HMM)

  -model 3: dependency parsing and spaCy

  -model 4: 



- model1 consists of the following steps:

  -Pre-processing of given data
  
  -Vector space model to represent documents (bag-of-words) then  Visualizing them (matlibplot) 
  
  -term frequency–inverse(=tf-idf) for weighting the word vectors, the more informative word vectors the higher weight, 
  
  -classification via k-NN(k-nearest neighbors algorithm) for predicting which category the documents belong to. 
  
  -at the end, we test and  evaluate the model
  
 


- model 2 consists of the following steps:

  -Logistic Regresjon for identifying which language a word belongs to via its phonetic features (transcriptions) 
    -training, predicting, evaluatiing, l Analysing of the model

  -Sequence models for association words to their phonetic features
  
    -Named Entity Recognition (=NER)
    
    -Hidden Markov Model (=HMM) associating Each word with a particular class, 
      using so-called BIO annotation

 
     
- model 3:

  -dependency parsing on Norwegian languauge with  CoNLL-U format.

  -First a transition-based parsing algorithm for dependency parsing

  -Then using NLP library spaCy to train a dependency parser on a Norwegian Treebank and evaluate the quality of the parser.

  -At the end a closer look at how the parser works on others variants of Norwegian language.



 Obs. more details followed  in the folders (written in Norwegian).
 
