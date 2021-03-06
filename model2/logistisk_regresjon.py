# -*- coding: utf-8 -*-

import urllib.request
import pandas, re, random
import numpy as np
import sklearn.linear_model, sklearn.metrics, sklearn.model_selection

ORDFILER = {"norsk":"https://github.com/open-dict-data/ipa-dict/blob/master/data/nb.txt?raw=true",
        "arabisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ar.txt?raw=true",
        "finsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fi.txt?raw=true",
        "patwa":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/jam.txt?raw=true",
        "farsi":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fa.txt?raw=true",
        "tysk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/de.txt?raw=true",
        "engelsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt?raw=true",
        "rumensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ro.txt?raw=true",
        "khmer":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/km.txt?raw=true",
        "fransk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fr_FR.txt?raw=true",
        "japansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ja.txt?raw=true",
        "spansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/es_ES.txt?raw=true",
         "svensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sv.txt?raw?true",
         "koreansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ko.txt?raw?true",
         "swahilisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sw.txt?raw?true",
         "vietnamesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/vi_C.txt?raw?true",
        "mandarin":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/zh_hans.txt?raw?true",
        "malayisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ma.txt?raw?true",
        "kantonesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/yue.txt?raw?true",
         "islandsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/is.txt?raw=true"}





class LanguageIdentifier:
    """Logistisk regresjonsmodell som tar IPA transkripsjoner av ord som input,
    og predikerer hvilke spr??kene disse ordene h??rer til."""

    def __init__(self):
        """Initialiser modellen"""
        # selve regresjonsmodellen (som brukes all CPU-er p?? maskinen for trening)
        #self.model = sklearn.linear_model.LogisticRegression(solver="lbfgs", multi_class='ovr', n_jobs=4)
        self.AlleUniqueSymboler_liste= []
        self.mapping={}
        self.model = sklearn.linear_model.SGDClassifier(loss="log", n_jobs=-1)

#.............hjelpe metoder ...........................................................
    def IPA_liste(self, transkripsjoner): #list over alle sympoler i triningssettet
        listeOverAlleSymboler=[]
        for transcpt in transkripsjoner:
            transkripsjon= list(transcpt)
            for symbol in transkripsjon:
                  listeOverAlleSymboler.append(symbol)
        return listeOverAlleSymboler

    def IPA_tupel(self,IPA_liste):  #list over alle unique sympoler i triningssettet (ingen forkomester)
        AlleUniqueSymboler_liste=list(set(IPA_liste))
        return AlleUniqueSymboler_liste;
#........................................................................................


    def extract_unique_symbols(self, transkripsjoner, min_nb_occurrences=10):
        AlleSymboler=self.IPA_liste(transkripsjoner)
        AlleUniqueSymboler=self.IPA_tupel(AlleSymboler)
        AlleUniqSymboler_minst10Forkom=[]

        for s1 in AlleUniqueSymboler:
            teller=0
            for s2 in AlleSymboler:
                if s1==s2:
                    teller+=1
                if teller==min_nb_occurrences: # erstette med min_nb_occurrences
                    AlleUniqSymboler_minst10Forkom.append(s1)
                    teller=0
                    break
        # print("lengde av alle sympoler (med forkomester) i trainingssettet er: ", len( AlleSymboler) )
        # print("lengde av alle uniqe sympoler (uten forkomester) i trainingssettet er: ", len(AlleUniqueSymboler) )
        # print("lengde av alle uniqe sympoler i trainingssettet, som forekommer mint 10 ganger, er: ", len(AlleUniqSymboler_minst10Forkom))
        return AlleUniqSymboler_minst10Forkom




    def extract_feats(self, transcriptions):

        matrise= np.zeros((len(transcriptions), len(self.AlleUniqueSymboler_liste)), dtype="int8") #np.zeros(rad, kal)

        kal=0
        while kal<len( self.AlleUniqueSymboler_liste):

            rad=0
            while rad < len(transcriptions):
                tran=[char for char in transcriptions[rad]] #dele hver transcpt til en liste av chars
                for bit in tran:
                    if bit==self.AlleUniqueSymboler_liste[kal]:
                        matrise[rad, kal]=1
                        break;
                rad+=1
            kal+=1

        # print("matrisen lagd i extract_feats() \n", matrise)
        return matrise



    def train(self, transcriptions, languages):
        self.AlleUniqueSymboler_liste=self.extract_unique_symbols(transcriptions,10)
        self.mapping = dict([(ettspr??k,heltall) for heltall,ettspr??k in enumerate(set(sorted(languages)))]) #gj??re mapping mellom spr??k og heltall
        spr??k_som_heltall=[self.mapping[heltall] for heltall in languages]
        if len(transcriptions)!= len(languages):
            raise ValueError("lengde av transripsjoner er ikke alike spr??k's")

        return self.model.fit(self.extract_feats(transcriptions), spr??k_som_heltall)



    def predict(self, transcriptions):

        predicted_tall=self.model.predict(self.extract_feats(transcriptions)) #predikere IDene til spr??k
        predicted_spraak=[]

        #favndling  tilbake til spr??knavnene
        for i in range(len(predicted_tall)):

            for key, value in self.mapping.items():
                if predicted_tall[i] == value:
                   predicted_spraak.append(key)
                   break
        # print("gjettede spr??k: ", predicted_spraak)
        return predicted_spraak




    def evaluate(self, transcriptions, languages):
        print("..............evaluering...............")
        predicted_spr??k= self.predict(transcriptions)
        accuracy_score=sklearn.metrics.accuracy_score(languages, predicted_spr??k)
        print("accuracy  score: ", accuracy_score)

        precision_score=sklearn.metrics.precision_score(languages, predicted_spr??k, average=None)
        recall_score=sklearn.metrics.recall_score(languages,predicted_spr??k,average=None)
        f1_score=sklearn.metrics.f1_score(languages,predicted_spr??k,average=None)


        orderedSpr??k={k: v for k, v in sorted(self.mapping.items(), key=lambda item: item[1])}
        print("orderedSpr??k",orderedSpr??k)

        pd=pandas.DataFrame([ precision_score,recall_score,f1_score] ) #,,columns=orderedSpr??k.keys columns=self.mapping["key"]
        #gi navn til rader
        pd.index=[ "precision_score", "recall_score","f1_score"]
        print(pd) #col ==spr??k  X   rader ==score

        f1_score_micro=sklearn.metrics.f1_score(languages,predicted_spr??k,average='micro')
        print("f1_score_micro is : ",f1_score_micro )
        f1_score_macro=sklearn.metrics.f1_score(languages,predicted_spr??k,average='macro')
        print("f1_score_macro is : ",f1_score_macro )


        #analyse av modellen


#.......en metode som finne ut h??yest og lavest trekk p?? norsk..........................
    def analyse_av_modellen(self):
        print("..............analyse av modellen...............")
        norsk_ID = self.mapping['norsk']
        norsk_trekk=self.model.coef_[norsk_ID]

        temp_h??yst=-1000
        temp_minst=1000
        indeksOfH??yest=0
        indeksOfmist=0
        for i in range(len(norsk_trekk)):
            if norsk_trekk[i]>temp_h??yst:
                temp_h??yst=norsk_trekk[i]
                indeksOfH??yest=i
            if norsk_trekk[i]<temp_minst:
                temp_minst=norsk_trekk[i]
                indeksOfmist=i

        print("h??yst_trekk p?? norsk er: ", self.AlleUniqueSymboler_liste[indeksOfH??yest])
        print("minst_trekk p?? norsk er: ", self.AlleUniqueSymboler_liste[indeksOfmist])


def extract_wordlist(max_nb_words_per_language=20000):
    full_wordlist = []
    for lang, wordfile in ORDFILER.items():

        print("Nedlasting av ordisten for", lang, end="... ")
        data = urllib.request.urlopen(wordfile)

        wordlist_for_language = []
        for linje in data:
            linje = linje.decode("utf8").rstrip("\n")
            word, transcription = linje.split("\t")

            # Noen transkripsjoner har feil tegn for "primary stress"
            transcription = transcription.replace("\'", "??")

            # vi tar den f??rste transkripsjon (hvis det finnes flere)
            # og fjerner slashtegnene ved start og slutten
            match = re.match("/(.+?)/", transcription)
            if not match:
                continue
            transcription = match.group(1)
            wordlist_for_language.append({"ord":word, "IPA":transcription, "spr??k":lang})
        data.close()

        # Vi blander sammen ordene, og reduserer mengder hvis listen er for lang
        random.shuffle(wordlist_for_language)
        wordlist_for_language = wordlist_for_language[:max_nb_words_per_language]

        full_wordlist += wordlist_for_language
        print("ferdig!")



    # N?? bygger vi en DataFrame med alle ordene
    full_wordlist = pandas.DataFrame.from_records(full_wordlist)

    # Og vi blander sammen ordene i tilfeldig rekkef??lge
    full_wordlist = full_wordlist.sample(frac=1)

    # Lage et treningssett og en testsett (med 10% av data)
    wordlist_train, wordlist_test = sklearn.model_selection.train_test_split(full_wordlist, test_size=0.1)
    print("Treningsett: %i eksempler, testsett: %i eksempler"%(len(wordlist_train), len(wordlist_test)))

    return wordlist_train, wordlist_test





######################
#Brukseksempel:
######################


if __name__ == "__main__":

    # Vi laster ned dataene (vi trenger kun ?? gj??re det ??n gang)
    train_data, test_data = extract_wordlist()


    # Vi teller antall ord per spr??k
    print("Statistikk over spr??kene i treningsett:")
    print(train_data.spr??k.value_counts())
    print("F??rste 30 ord:")
    print(train_data[:30])
    #
    # Vi bygge og trene modellen
    model = LanguageIdentifier()
    transcriptions = train_data.IPA.values
    languages = train_data.spr??k.values


    model.train(transcriptions, languages)
    # Vi kan n?? test modellen p?? nye data
    predicted_langs = model.predict(["konstitu????jon","??r??nl????", "stjourtnar??skrau????n", "b??nd??sv????fasz????"]) #
    print("Mest sansynnlige spr??k for ordene:",predicted_langs)

    # # Til slutt kan vi evaluere hvor godt modellen fungerer p?? testsett
    model.evaluate(test_data.IPA.values, test_data.spr??k.values)


#..........................analyse_av_modellen.............................................
    v_lyd_norsk=0
    v_lyd_ikke_norsk=0
    for i in range(len(languages)):

        if languages[i] =="norsk":
            for trekk in transcriptions[i]:
                if trekk=='??':
                    v_lyd_norsk+=1
        else:
            for trekk in transcriptions[i]:
                if languages[i] =="norsk":
                    continue
                if trekk=='??':
                    v_lyd_ikke_norsk+=1

    model.analyse_av_modellen()
    print("total antall ganger ??  kommer med norske ord med",v_lyd_norsk,"ganger vs. ikke norske ord med ", v_lyd_ikke_norsk," ganger.")
