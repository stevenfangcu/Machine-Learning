import re
import numpy as np
import pandas as pd
from pprint import pprint
#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
#lemmatizer
from nltk.stem import WordNetLemmatizer
#LDA tools
import pyLDAvis
import pyLDAvis.gensim
from nltk.corpus import stopwords
stopwords = ['i','me','my','every','myself','we','our','ours', 'ourselves', 'you','I','Use','use','Have','CJOC','cjoc','Canada','canada','fit','of','also' 'your','yours','yourself','yourselves','he','him','his','himself','she','her','herself','it','its',"it's",'itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','The','A','under','again','further','would','then','once','here','there','when','where','why','how','all','any','-','s',"'s'",'w','',' ','both','each',"aren't",'few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just',"don't",'should','now']

def main():
    #df = pd.read_csv('trainingSet.csv', encoding="cp1252")
    df = pd.read_csv('ImpactCaseStudiesCsv.csv', encoding="cp1252")
    #print(df.ClusterTerms.unique())
    topicModeling(df)
    df.head()
    pass
def topicModeling(df):
    #data = df.CleanedText.values.tolist() used on rimpac
    data = df.CaseStudy.values.tolist()
    data_words = list(tokenize(data))
    bigram = Phrases(data_words,min_count=5,threshold=100)
    trigram = Phrases(bigram[data_words],threshold=100)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    cleanedData = removeStopwords(data_words)
    #make bigrams
    make_bigrams = [bigram_mod[doc] for doc in cleanedData]
    make_trigrams = [trigram_mod[bigram_mod[doc]] for doc in cleanedData]
    #dictionary
    id2word = corpora.Dictionary(cleanedData)
    #corpus
    corpus = cleanedData
    #tfidf
    tfidf = [id2word.doc2bow(text) for text in corpus]
    #pprint([[(id2word[id],freq) for id, freq in cp] for cp in tfidf]) #prints frequency of each term
    lda_model = gensim.models.ldamodel.LdaModel(corpus=tfidf,id2word=id2word,num_topics=13,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
    pprint(lda_model.print_topics())
    #compute perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(tfidf))
    #compute coherence
    coherence_model_lda = CoherenceModel(model=lda_model,texts=cleanedData, dictionary=id2word,coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    print('\nCoherence: ',coherence_score)
    #topic prediction
    unseenText = [
    ['danielo','ll','wallet','resturant','mcDonalds'],
    ['Paper','observations','benefits','army','air','force'],
    ['representation','support','execrises','random','forest','charger']
    ]
    unseen_bigrams = [bigram_mod[doc] for doc in unseenText]
    unseen_trigrams = [trigram_mod[bigram_mod[doc]] for doc in unseenText]
    test_tfidf = [id2word.doc2bow(text) for text in unseenText]
    for index in test_tfidf: #printing off predicitions of the topics
        sortprint = sorted(lda_model.get_document_topics(index),key=lambda x: x[1],reverse=True)
        pprint(sortprint) # top down
    pass
def tokenize(textData): #tokenizing the data
    for sentence in textData:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))
def removeStopwords(texts): #removing stopwords
    returnData = []
    for para in texts:
        transferData = []
        for word in para:
            if word not in stopwords:
                transferData.append(word)
        returnData.append(transferData)
    return returnData
if __name__ == "__main__":
    main()
