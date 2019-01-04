import nltk, sys

#Targets
targets_ = open(sys.argv[1]).readlines()
targets = []
for sent in targets_:
    try:
        sent = nltk.word_tokenize(sent)
    except:
        sent = sent.split(" ")
    targets.append([sent]) 

#Outputs
outputs_ = open(sys.argv[2]).readlines()
outputs = []
for sent in outputs_:
    try:
        sent = nltk.word_tokenize(sent)
    except:
        sent = sent.split(" ")
    outputs.append(sent) 
#print(targets, outputs)
#Get BLEU score between targets and outputs
print "BLEU : ", nltk.translate.bleu_score.corpus_bleu(targets[:len(outputs)], outputs)
