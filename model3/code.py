import spacy
from in2110.conllu import ConlluDoc
nb = spacy.load("model/model-best")
conllu_dev = ConlluDoc.from_file("no_bokmaal-ud-dev.conllu")

def attachment_score(true, pred):

    num_words = 0
    for doc in true:
        for token in doc:
            num_words += 1

    parsed = 0
    for doc in pred:
        if doc.is_parsed:
            parsed += 1
    #print("attachment_score: called with {} gt docs and {} parsed predictions".format(len(true),parsed)) #DEBUG
    # count correctly parsed docs
    words_with_correct_head = 0
    words_with_correct_head_and_deprel = 0


    for t,p in zip(true, pred):
        for token_t, token_p in zip(t,p):
            if token_t.head.text == token_p.head.text:
                words_with_correct_head += 1
                if token_t.dep_ == token_p.dep_:
                    words_with_correct_head_and_deprel += 1


    # uas: unlabeled attachment score
    # words with correct head / words
    uas = words_with_correct_head / num_words

    # las: labeled attachment score
    # words with correct head and deprel / words
    las = words_with_correct_head_and_deprel / num_words

    return uas, las

print("converting from conllu to spacy")
dev_docs = conllu_dev.to_spacy(nb) # ground truth
dev_docs_unlabeled = conllu_dev.to_spacy(nb, keep_labels=False) # to become predictions

nynorsk_dev = ConlluDoc.from_file("no_nynorsk-ud-dev.conllu")
nynorsk_docs = nynorsk_dev.to_spacy(nb) # ground truth
nynorsk_docs_unlabeled = nynorsk_dev.to_spacy(nb, keep_labels=False)

nynorsklia_dev = ConlluDoc.from_file("no_nynorsklia-ud-dev.conllu")
nynorsklia_docs = nynorsklia_dev.to_spacy(nb) # ground truth
nynorsklia_docs_unlabeled = nynorsklia_dev.to_spacy(nb, keep_labels=False)

predictions = []

#print(dev_docs_unlabeled[0])
print("parsing...")
for doc in dev_docs_unlabeled:
    nb.parser(doc)

for doc in nynorsk_docs_unlabeled:
    nb.parser(doc)

for doc in nynorsklia_docs_unlabeled:
    nb.parser(doc)

print("done")

uas,las = attachment_score(dev_docs, dev_docs_unlabeled)
print("(no_bokmaal-ud-dev)\tuas: {}\tlas: {}".format(uas,las))

uas,las = attachment_score(nynorsk_docs, nynorsk_docs_unlabeled)
print("(no_nynorsk-ud-dev)\tuas: {}\tlas: {}".format(uas,las))

uas,las = attachment_score(nynorsklia_docs, nynorsklia_docs_unlabeled)
print("(no_nynorsklia-ud-dev)\tuas: {}\tlas: {}".format(uas,las))
