import re

class NamedEntityRecogniser:

    def __init__(self):


        # alle labellene som forekommer i treningsettet
        self.labels = set("O")

        # alle token som forekommer i treningsettet
        self.vocab = set()

        # hvor mange ganger en label (f.eks. B-ORG) forekommer i treningsettet
        self.label_counts = {"O":1}

        # hvor mange transisjoner fra label_1 til label2 forekommer i treningsettet
        self.transition_counts = {}

        # hvor mange emisjoner fra label til token forekommer i treningsettet
        self.emission_counts = {("O", "<UNK>"):1}

        # Sansynnlighet P(label_2 | label_1)
        self.transition_probs = {}

        # Sansynnlighet P(token | label)
        self.emission_probs = {}


    def fit(self, tagged_text):

        # Ekstrahere setninger og navngitte enheter markert i hver setning
        sentences, all_spans = preprocess(tagged_text)

        for sentence, spans in zip(sentences, all_spans):

            # Ekstrahere labelsekvenser, med BIO (også kalt IOB) marking
            label_sequence = get_BIO_sequence(spans, len(sentence))

            # Oppdatere tallene
            self._add_counts(sentence, label_sequence)

        # Beregne sansynnlighetene (transition og emission) ut fra tallene
        self._fill_probs()


#...........hjelpe metode....................
#sjekke om et label finnes i labels

    def _ErFunnet(self, ordbok, nøkkelord):
        for key, value in ordbok.items():
            if key==nøkkelord:
                return True
        return False
#...........................................

    def _add_counts(self, sentence, label_sequence):
        #kaste untakk hvis sentence og label_sequence ikke har like lengde

        #legge alle ord i setningen til vocab
        for ord in sentence:
                self.vocab.add(ord)

        merkelapper_i_rekkefølge=["start"] #som temp hjelpeliste for å finne transisjoner

        for i in range(len(label_sequence)):
                if i==0:
                    if self._ErFunnet(self.label_counts,"start"): #sjekke om merkelapp "start" er i label_counts
                        self.label_counts["start"]+=1 #hvis ja, plusse den med 1
                    else:
                        self.label_counts["start"]=1 #ellers opprette den og  tilordne med 1

                if label_sequence[i].startswith('B'): #sjekke om det er en B-merkelapp i label_sequence (obs. jeg kunne  bruke sentence for å sjekke merkelpper isteden)
                    self.labels.add(label_sequence[i])   #legge den tabels (obs. labels tillater ikke forekommester.)
                    merkelapper_i_rekkefølge.append(label_sequence[i]) #legge merkelappen til listen slik at jeg kan bruke i transisjoner senere

                    if self._ErFunnet(self.label_counts, label_sequence[i]):#sjekk om  merkelappen er i label_counts
                        self.label_counts[label_sequence[i]]+=1 #hvis ja bare plusse den med 1
                    else:
                        self.label_counts[label_sequence[i]]=1  #ellers oprette den og gi den verdi 1 å starte med

                    #emisjoner
                    emisjon= (label_sequence[i], sentence[i]) # en temp variabel som tar vare på  B-merkelapp, sammen med ord den beskriver
                    if self._ErFunnet(self.emission_counts, emisjon): #sjekke hvis emisjonen er med på emission_counts
                        self.emission_counts[emisjon]+=1 #Hvis ja, bare plusse med en
                    else:
                        self.emission_counts[emisjon]=1#ellers opprette den og gi den verdi 1 å starte med

                #Her er det en annen hovedsjekk. Den sjekker I-tag
                if label_sequence[i].startswith('I'):
                    emisjon= (label_sequence[i], sentence[i]) #en temp variabel som tar vare på  B-merkelapp, sammen med ord den beskriver
                    if self._ErFunnet(self.emission_counts, emisjon):#sjekke hvis emisjonen er med på emission_counts
                        self.emission_counts[emisjon]+=1#Hvis ja, bare plusse med en
                    else:
                        self.emission_counts[emisjon]=1#ellers opprette den og gi den verdi 1 å starte med

                    #legge merkelpper som har  prefiks 'I'  til labels
                    if self._ErFunnet(self.label_counts, label_sequence[i]):  #sjekke hvis merkelappen er med på label_counts
                        self.label_counts[label_sequence[i]]+=1 #Hvis ja, bare plusse med en
                    else:
                        self.label_counts[label_sequence[i]]=1 #ellers opprette den og gi den verdi 1 å starte med

        # transisjoner
        if len(merkelapper_i_rekkefølge)>1: #hvis det er  en label (i tillegg til "start") liste merkelapper_i_rekkefølge
            j=1
            while j <len(merkelapper_i_rekkefølge): #sjekke om listen har minst en label + start
                transisjon= (merkelapper_i_rekkefølge[j-1], merkelapper_i_rekkefølge[j]) # Merk at jeg begynner med -1 slik at jeg ikke får "Out_of_Ramge exception" i slutten av listen
                if self._ErFunnet(self.transition_counts, transisjon): #hvis transisjon /som er av type tupel) er med på transition_counts
                    self.transition_counts[transisjon]+=1 #ja, plusse med 1
                else:
                    self.transition_counts[transisjon]=1#ellers opprette den med verdi 1
                j+=1


    def _fill_probs(self, alpha_smoothing=1E-6):

        for key, value in self.transition_counts.items():
            self.transition_probs[key] = value/self.label_counts[key[0]]
        #en annen tilnærmering
        # transition_total = 0
        # for transition in self.transition_counts:
        #     transition_total+= self.transition_counts[transition]
        #
        # for (key, value) in self.transition_counts.items():
        #     self.transition_probs [key] = value/transition_total

        for key, value in self.emission_counts.items():
            self.emission_probs[key]= (value+alpha_smoothing)/self.label_counts[key[0]]+(len(self.vocab)*alpha_smoothing)

        return self.transition_counts, self.emission_probs


    def _viterbi(self, sentence):
        """Kjører Viterbi-algoritmen på setningen (liste over tokens), og
        returnerer to outputs:
        1) en labelsekvens (som har samme lengde som setningen)
        2) sansynnlighet for hele sekvensen """

        # De 2 datastrukturer fra Viterbi algoritmen, som dere må fylle ut
        lattice = [{label:None for label in self.labels}
                           for _ in range(len(sentence))]
        backpointers = [{label:None for label in self.labels}
                        for _ in range(len(sentence))]

        # Fylle ut lattice og backpointers for setningen
        for i, token in enumerate(sentence):
            for label in self.labels:
                raise NotImplementedException()

        # Finne ut det mest sannsynlig merkelapp for det siste ordet
        best_final_label = max(lattice[-1].keys(), key=lambda x: lattice[-1][x])
        best_final_prob = lattice[-1][best_final_label]

        # Ekstrahere hele sekvensen ved å følge de "backpointers"
        best_path = [best_final_label]
        for i in range(i,0,-1):
            best_path.insert(0, backpointers[i][best_path[0]])

        # Returnerer den mest sannsynlige sekvensen (og dets sannsynlighet)
        return best_path, best_final_prob



    def label(self, text):
        """Gitt en tokenisert tekst, finner ut navngitte enheter og markere disse
        med XML tags. """
        sentences, _ = preprocess(text)
        spans = []
        for sentence in sentences:
            sentence = [token if token in self.vocab else "<UNK>" for token in sentence]
            label_sequence, _ = self._viterbi(sentence)
            spans.append(get_spans(label_sequence))

        return postprocess(sentences, spans)


    #..................hjelpemetode..................
    def skrivUt(self):
            print("Antall ord  ",len(self.vocab),"\nlabels  \n",self.labels,"\n label_counts \n " ,self.label_counts,"\n transition_counts  \n" ,self.transition_counts, "\n emisjoner_counts  \n",self.emission_counts)
            print("\n transition_probs\n  ",self.transition_probs,"\n emission_probs \n " ,self.emission_probs)


def get_BIO_sequence(spans, sentence_length):
    bio_markeringListe=[]

    #starte med å fylle listen med med 'o'-er
    for i in range(sentence_length):
        bio_markeringListe.append('O')

    for tupel in spans:
        bio_markeringListe[tupel[0]]="B-"+tupel[2]

        i=tupel[0]+1 #hopper over B-"tag", fordi jeg har nettopp lagt til listen
        j=tupel[1]
        #hvis det er I-"tag" i spans
        while i < j:
                bio_markeringListe[i]="I-"+tupel[2]
                i+=1
    return bio_markeringListe


def get_spans(label_sequence):
    """Gitt en labelsekvens med BIO markering, returner en lister over "spans" med
    navngitte enheter. Metoden er altså den motsatte av get_BIO_sequence"""

    spans = []
    i = 0
    while i < len(label_sequence):
        label = label_sequence[i]
        if label.startswith("B-"):
            start = i
            label = label[2:]
            end = start + 1
            while end < len(label_sequence) and label_sequence[end].startswith("I-%s"%label):
                end += 1
            spans.append((start, end, label))
            i = end
        else:
            i += 1
    return spans


def preprocess(tagged_text):
    """Tar en tokenisert tekst med XML tags (som f.eks. <ORG>Stortinget</ORG>) og
    returnerer en liste over setninger (som selv er lister over tokens), sammen med
    en liste av samme lengde som inneholder de markerte navngitte enhetene. """

    sentences = []
    spans = []

    for i, line in enumerate(tagged_text.split("\n")):

        tokens = []
        spans_in_sentence = []

        for j, token in enumerate(line.split(" ")):

            # Hvis token starter med en XML tag
            start_match = re.match("<(\w+?)>", token)
            if start_match:
                new_span = (j, None, start_match.group(1))
                spans_in_sentence.append(new_span)
                token = token[start_match.end(0):]

            # Hvis token slutter med en XML tag
            end_match = re.match("(.+)</(\w+?)>$", token)
            if end_match:
                if not spans_in_sentence or spans_in_sentence[-1][1]!=None:
                    raise RuntimeError("Closing tag without corresponding open tag")
                start, _ , tag = spans_in_sentence[-1]
                if tag != end_match.group(2):
                    raise RuntimeError("Closing tag does not correspond to open tag")
                token = token[:end_match.end(1)]
                spans_in_sentence[-1] = (start, j+1, tag)

            tokens.append(token)

        sentences.append(tokens)
        spans.append(spans_in_sentence)

    return sentences, spans


def postprocess(sentences, spans):
    """Gitt en liste over setninger og en tilsvarende liste over "spans" med
    navngitte enheter, produserer en tekst med XML markering."""

    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        new_sentence = list(sentence)
        for start, end, tag in spans[i]:
            new_sentence[start] = "<%s>%s"%(tag, new_sentence[start])


            new_sentence[end-1] = "%s</%s>"%(new_sentence[end-1], tag)
        tagged_sentences.append(" ".join(new_sentence))

    return "\n".join(tagged_sentences)

#....................main..........................

f = open("norne_train.txt", "r")
# f = open("norne_test.txt", "r")
content=f.read()

obj= NamedEntityRecogniser()
obj.fit(content)
obj.skrivUt()
