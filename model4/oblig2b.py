# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The file contains two implementations:
1) A method to compute BLEU scores
2) A retrieval-based chatbot based on TF-IDF.
"""


import collections
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def get_sentences(text_file):
    """Given a text file with one (tokenised) sentence per line, returns a list
    of sentences , where each sentence is itself represented as a list of tokens.
    The tokens are all converted into lowercase.
    """

    sentences = []
    fd = open(text_file)
    for sentence_line in fd:

        # We convert everything to lowercase
        sentence_line = sentence_line.rstrip("\n").lower()

        # We also replace special &apos; characters with '
        sentence_line = sentence_line.replace("&apos;", "'")

        sentences.append(sentence_line.split())
    fd.close()

    return sentences


def get_ngrams(sentence, ngram_order):
# gets ngrams for one sentence
    length = len(sentence)

    sentence_ngrams = []
    for word_index in range(length):
        if (length-word_index < ngram_order): # can't capture more ngrams
            break

        # capture N words beginning with sentence[word_index] and ending with sentence[ngram_end]
        # thus, word_index and ngram_end becomes a sliding window of n word inidces
        # following the whole sentence and capturing all ngrams of the given order

        ngram_start = word_index
        ngram_end = word_index + ngram_order -1
        #print("start:{} end:{}".format(ngram_start,ngram_end))

        # adding ngram-part 0
        ngram = sentence[ngram_start]
        i = ngram_start+1

        #appending the rest of the ngram-parts
        while i <= ngram_end:
            ngram = ngram +" "+ sentence[i]
            i += 1
        # we have collected an ngram and increment before collecting the next
        ngram_start += 1
        ngram_end +=1
        sentence_ngrams.append(ngram)
    #assert(length - (ngram_order-1) == len(sentence_ngrams))
    if max((length - ngram_order) +1,0) != len(sentence_ngrams):
        print("assert failed: ")
        print("max((length - ngram_order) +1,0) != len(sentence_ngrams): {} - {} +1 != {}".format(length, ngram_order, len(sentence_ngrams)))
        return None
    return sentence_ngrams


def compute_precision(reference_file, output_file, ngram_order):
    """
    Computes the precision score for a given N-gram order. The first file contains the
    reference translations, while the second file contains the translations actually
    produced by the system. ngram_order is 1 to compute the precision over unigrams,
    2 for the precision over bigrams, and so forth.
    """

    ref_sentences = get_sentences(reference_file) # ground truth
    output_sentences = get_sentences(output_file) # sample

    num_correct_total = 0
    num_ngrams_total = 0

# output = get_sentences(output_file)
# len_output = 0
# for sentence in output:
#     for word in sentence:
#         len_output += 1

    for ref_single_sentence,output_single_sentence in zip(ref_sentences, output_sentences):
        num_correct = 0
        #num_ngrams_in_sentence = 0
        iteration = 0

        reference_ngrams = get_ngrams(ref_single_sentence, ngram_order)
        output_ngrams = get_ngrams(output_single_sentence, ngram_order)
        num_ngrams_in_ref_sentence = len(reference_ngrams)
        num_ngrams_in_output_sentence = len(output_ngrams)
        for ngram_reference in reference_ngrams:
            for ngram_output in output_ngrams:
                if ngram_reference == ngram_output:
                    num_correct += 1

        num_correct_total += num_correct
        num_ngrams_total += num_ngrams_in_output_sentence

    precision = num_correct_total/num_ngrams_total
    #             x / antall ngrams i output-setninger ( ikke i fasit-setninger)
    print("precision({}): {}/{} = {}".format(ngram_order, num_correct_total, num_ngrams_total, precision))
    return precision


def compute_brevity_penalty(reference_file, output_file):
    """Computes the brevity penalty."""
    """ brevity penalty = min( 1, antall ord i systemets setninger / antall ord i fasitens setninger  ) """
    ref_sentences = get_sentences(reference_file)
    output_sentences = get_sentences(output_file)

    ref_wordcount = 0
    output_wordcount = 0
    for ref_sentence,output_sentence in zip(ref_sentences,output_sentences):
        ref_wordcount += len(ref_sentence)
        output_wordcount += len(output_sentence)

    print("antall ord i {}:{}\nantall ord i {}:{}".format(reference_file,ref_wordcount,output_file,output_wordcount))
    quotient = output_wordcount/ref_wordcount
    print("brevity penalty:{}".format(quotient))
    if quotient < 1:
        return quotient
    else:
        return 1


def compute_bleu(reference_file, output_file, max_order=4):
    """
    Given a reference file, an output file from the translation system, and a
    maximum order for the N-grams, computes the BLEU score for the translations
    in the output file.
    """

    precision_product = 1
    for i in range(1, max_order+1):
        precision_product *= compute_precision(reference_file, output_file, i)

    brevity_penalty = compute_brevity_penalty(reference_file, output_file)

    bleu = brevity_penalty * math.pow(precision_product, 1/max_order)
    return bleu




class RetrievalChatbot:
    """Retrieval-based chatbot using TF-IDF vectors"""

    def __init__(self, dialogue_file):
        """Given a corpus of dialoge utterances (one per line), computes the
        document frequencies and TF-IDF vectors for each utterance"""

        # We store all utterances (as lists of lowercased tokens)
        self.utterances = []
        fd = open(dialogue_file)
        for line in fd:
            utterance = self._tokenise(line.rstrip("\n"))
            self.utterances.append(utterance)
        fd.close()

        self.doc_freqs = self._compute_doc_frequencies()
        self.tf_idfs = [self.get_tf_idf(utterance) for utterance in self.utterances]


    def _tokenise(self, utterance):
        """Convert an utterance to lowercase and tokenise it by splitting on space"""
        return utterance.strip().lower().split()


    def _compute_doc_frequencies(self):
        """Compute the document frequencies (necessary for IDF)"""

        doc_freqs = {}
        for utterance in self.utterances:
            for word in set(utterance):
                doc_freqs[word] = doc_freqs.get(word, 0) + 1
        return doc_freqs



    def get_tf_idf(self, utterance):
        """Compute the TF-IDF vector of an utterance. The vector can be represented
        as a dictionary mapping words to TF-IDF scores. The utterance is a list of
        (lowercased) tokens. """

        dict_tfidf = {}
        for word in utterance:
            TF = 0
            for w in utterance:
                if word == w:
                    TF += 1
            N = len(self.utterances)

            No = 0
            for u in self.utterances:
                for w in u:
                    if w == word:
                        No += 1

            if No == 0:
                No = 0.0000000000000001
            IDF = math.log(N/No,10)
            dict_tfidf[word] = TF * IDF
        return dict_tfidf


    def _get_norm(self, tf_idf):
        """Compute the vector norm"""

        return math.sqrt(sum([v**2 for v in tf_idf.values()]))


    def get_response(self, query):
        """
        Finds out the utterance in the corpus that is closed to the query
        (based on cosine similarity with TF-IDF vectors) and returns the
        utterance following it.
        """

        # If the query is a string, we first tokenise it
        if type(query)==str:
            query = self._tokenise(query)


        query_tfidf = self.get_tf_idf(query)
        # print("query tfidf: {}".format(query_tfidf))
        best_similarity = -1
        most_similar_string = None

        index = 0

        for tfidf in self.tf_idfs:
            similarity = self.compute_cosine(query_tfidf, tfidf)
            if similarity > best_similarity:
                # print("new best similarity: {}".format(similarity))
                # print("new best index: {}".format(index))
                best_similarity = similarity
                best_index = index
            index += 1
        if best_index+1 == len(self.utterances):
            return self.utterances[0] # wrap-around
        return self.utterances[best_index+1]


    def compute_cosine(self, tf_idf1, tf_idf2):
        """Computes the cosine similarity between two vectors"""
        #argmax(|c|,1)* dot(qT,ti)/norm(q) * norm(ti)

        a = np.array(list(tf_idf1.values()))
        b = np.array(list(tf_idf2.values()))

        if len(a) >= len(b):
            #pad b
            diff = len(a)-len(b)
            b = np.pad(b,(0,diff),mode='constant')
        else:
            #pad a
            diff = len(b)-len(a)
            a = np.pad(a,(0,diff),mode='constant')

        dot_product = np.vdot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine_similarity = dot_product / (norm_a * norm_b)

        #print("cosine similarity:\n{}".format(cosine_similarity))
        return cosine_similarity

def ngram_count(sentences,n):
    count = 0
    for s in sentences:
        count += (len(s)-n)+1
    return count

filename_fasit = "lotr.en"
filename_reference = "lotr.out_without_dic.en"

fasit = get_sentences(filename_fasit)
ngc = []
for i in range(1,5):
    ngc.append(ngram_count(fasit,i))

print("n-gram i {}:".format(filename_fasit))
for i in range(1,5):
    print("{}-gram: {}".format(i,ngc[i-1]))

print("\n")
len_fasit = 0
for sentence in fasit:
    for word in sentence:
        len_fasit += 1


#for i in range(1,5):
#    compute_precision(filename_fasit, filename_reference, i)

print("Uten phrase-table")
bleu = compute_bleu(filename_fasit, filename_reference)
print("BLEU: {}".format(bleu))
print("\n\n")
filename_reference = "lotr.out_with_dic.en"
print("Med (redigert) phrase-table:")
bleu = compute_bleu(filename_fasit, filename_reference)
print("BLEU: {}".format(bleu))

### chatbot ###
filename_chatbot = "lotr.en"
print("\nloading chatbot from {}...".format(filename_chatbot))
cb = RetrievalChatbot(filename_chatbot)
print("done! (press ctrl + C to exit)\n")

while True:
    print("\nchatbot>",end=' ')
    request = input()
    while request == '':
        print("chatbot>",end=' ')
        request = input()

    response = cb.get_response(request)
    [print(w,end=' ') for w in response]
