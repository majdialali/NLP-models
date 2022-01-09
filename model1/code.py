
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from in2110.corpora import norec
from in2110.oblig1 import scatter_plot


#......................hjelpebiblioteker som jeg har importert...................................

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# import pandas as pd
#...................................Oppgave 1a..........................................

data=[]
labels=[]


def prepare_data(documents):

    for d in documents:

        if d[1].get('category')== 'games' or d[1].get('category')=='literature' or d[1].get('category')=='restaurants':
            data.append(d[0])
            labels.append(d[1])

    return data, labels

prepare_data(norec.train_set()) #kall på metode




#...................................Oppgave 1b, første del..........................................
def tokenize(text):
    """Tar inn en streng med tekst og returnerer en liste med tokens."""
    return text.split()


ListeOverAllTokensiKorpuset=[]

for element in norec.train_set():
    hjelpeliste= tokenize(element[0])
    for token in hjelpeliste:
        ListeOverAllTokensiKorpuset.append(token)

ordtyper_ListeOverAllTokensiKorpuset = set(ListeOverAllTokensiKorpuset)


#.....test/print......

# print("antall tokens i hele korpuset: ",len(ListeOverAllTokensiKorpuset))  #via bruk av den orginale funksjon tokenize()
# print("antall oredtyper i hele korpuset: ",len(ordtyper_ListeOverAllTokensiKorpuset))

#...................................Oppgave 1b, andre  del..........................................


#splitte med nltk sin funksjon  word_tokenise()
antalltokens_wordTokanise=[]
for element in norec.train_set():
    hjelpeliste= word_tokenize(element[0])
    for token in hjelpeliste:
        antalltokens_wordTokanise.append(token)

ordtyper_wordTokanise = set(antalltokens_wordTokanise)

#.....test/print......

# print("antall tokens i nltk: ", len(antalltokens_wordTokanise))
# print("antall ordtyper i nltk: ", len(ordtyper_wordTokanise))


#...................................Oppgave 1b, tredje  del..........................................


#har endret på funksjonen tokenize() ved å gjøre alle bokstaver til små og å fjerne  punctuation marks og stopwords
def tokenize(text):
    """Tar inn en streng med tekst og returnerer en liste med tokens."""
    text = text.lower()   #gjøre til små bokstaver
    text= "".join([c for c in text if c not in string.punctuation]) #fjerne tegnsetting
    tolkens= word_tokenize(text)
    stop_words = set(stopwords.words('norwegian'))
    text=[word for word in tolkens if word not in stop_words]  #fjerne funksjonord

    return text


antalltokens_Tokanise_etterEndringer=[]
for element in norec.train_set():
    hjelpeliste= tokenize(element[0])
    for token in hjelpeliste:
        antalltokens_Tokanise_etterEndringer.append(token)

ordtyper_Tokanise_etterEndringer= set(antalltokens_Tokanise_etterEndringer)


#.....test/print......

# print("antall tokens etter mine endring i tokenise(): ", len(antalltokens_Tokanise_etterEndringer))
# print("antall ordtyper etter mine endring i tokenise(): ", len(ordtyper_Tokanise_etterEndringer))



#..................Oppgave 1 c første del.............................................



antall_dok_games=0
antall_dok_literature=0
antall_dok_restaurants=0
for d in labels:
        if d.get('category')== 'games': #or d[1].get('category')=='literature' or d[1].get('category')=='restaurants':
            antall_dok_games=antall_dok_games+1
        if d.get('category')== 'literature':
            antall_dok_literature=antall_dok_literature+1
        if d.get('category')== 'restaurants':
            antall_dok_restaurants=antall_dok_restaurants+1


#test

# print("antall_dok_games: ", antall_dok_games)
# print("antall_dok_literature: ", antall_dok_literature)
# print("antall_dok_restaurants: ", antall_dok_restaurants)




#....................................... .................................................
#..............................   OPPGAVE   2  ...........................................
#....................................... .................................................
#når du kjører koden,pass på at tokanise(), den jeg har gjord endringer på, ikke er kommentert.


#............oppgave 2, a ...............................


listeOver_deTreKategorier=[]
# prepare_data(norec.train_set()) #kall på metode

#liste over kategorien til hert dok i trainingdatsettet (gaming, literature og restaurants)

for dok in labels:
            listeOver_deTreKategorier.append(dok.get('category'))




class Vectorizer:

    def __init__(self):
        self.vectorizer =CountVectorizer(max_features=5000, analyzer=tokenize)
        self.tfidf = TfidfTransformer()

    listeAvOrdTyper5000=[]
    def vec_train(self, data):
        vec = None

        vec = self.vectorizer.fit(data)
        listeAvOrdTyper5000=vec.get_feature_names() #lagre listen av ordtyper for å bruke i test
        #pirnt(vec.vocabulary_)
        #print(vec.get_feature_names())   # bare for fit()  # hente en String liste med unike ord i alfabetisk rekkefølge
        vec= self.vectorizer.transform(data)
        vec_tfidf =self.tfidf.fit_transform(vec)

        return  vec_tfidf #vec  #  returnere en liste med vektorer med og uten tfidf




#.............................................................................
#..........................................oppgave 3..........................
#..........................................:::::: ............................
#................................a) knn.......................................
def create_knn_classifier(vec, labels, k):
    clf = KNeighborsClassifier(k)
    clf=clf.fit(vec, labels)
    return clf


train_data, train_labels = prepare_data(norec.train_set())
dev_data, dev_labels = prepare_data(norec.dev_set())
test_data, test_labels = prepare_data(norec.test_set())


# hente kategoriene
labels2=[]
for dok in train_labels:
            labels2.append(dok.get('category'))



#test
# v1 = Vectorizer()
# dokumentvectorer1=v1.vec_train(train_data)
#kall på metode
# print(create_knn_classifier(dokumentvectorer1,labels2 ,2))

#................................b) evaluering.......................................

v2 = Vectorizer()

dokumentvectorer2=v2.vec_train(dev_data)

#test
# kkassifikator_med_K1= create_knn_classifier(dokumentvectorer2,labels2 ,1)
# kkassifikator_med_K2= create_knn_classifier(dokumentvectorer2,labels2 ,2)
# kkassifikator_medK10= create_knn_classifier(dokumentvectorer2,labels2 ,100)
# kkassifikator_medK100= create_knn_classifier(dokumentvectorer2,labels2 ,100)
#
#
# predikasjoner_k1=kkassifikator_med_K1.predict(dokumentvectorer2)
# predikasjoner_k2=kkassifikator_med_K2.predict(dokumentvectorer2)
# predikasjoner_k10=kkassifikator_medK10.predict(dokumentvectorer2)
# predikasjoner_k100=kkassifikator_medK100.predict(dokumentvectorer2)
#
# print(accuracy_score(labels2,predikasjoner_k1))
# print(accuracy_score(labels2,predikasjoner_k2))
# print(accuracy_score(labels2,predikasjoner_k10))
# print(accuracy_score(labels2,predikasjoner_k100))


#................................c) testing.......................................
v3 = Vectorizer()

dokumentvectorer3=v2.vec_train(test_data)

#test
# kkassifikator_med_K1_t= create_knn_classifier(dokumentvectorer3,labels2 ,1)
# kkassifikator_med_K2_t= create_knn_classifier(dokumentvectorer3,labels2 ,2)
# kkassifikator_medK10= create_knn_classifier(dokumentvectorer3,labels2 ,100)
# kkassifikator_medK100= create_knn_classifier(dokumentvectorer3,labels2 ,100)
#
#
# predikasjoner_k1_t=kkassifikator_med_K1_t.predict(dokumentvectorer3)
# predikasjoner_k2_t=kkassifikator_med_K2_t.predict(dokumentvectorer3)
# predikasjoner_k10_t=kkassifikator_medK10.predict(dokumentvectorer3)
# predikasjoner_k100_t=kkassifikator_medK100.predict(dokumentvectorer3)
#
# print(accuracy_score(labels2,predikasjoner_k1_t))
# print(accuracy_score(labels2,predikasjoner_k2_t))
# print(accuracy_score(labels2,predikasjoner_k10_t))
# print(accuracy_score(labels2,predikasjoner_k100_t))
