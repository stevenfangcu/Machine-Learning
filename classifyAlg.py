#classes imported
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
import string
import re
from nltk.stem.snowball import SnowballStemmer
from objClasses import csvObj, Obj
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix, f1_score,accuracy_score,precision_recall_fscore_support
from IPython.display import display
import csv
import matplotlib.pyplot as plt
import sys,os
from sklearn.svm import SVC,SVR
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from nltk.corpus import wordnet as wn
#SVM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from gensim import models
from nltk.stem import WordNetLemmatizer
#globals
globalPunc = string.punctuation
#getting rid of some pubications that i would like to keep in the text
globalPunc = globalPunc.replace("-","")
globalPunc = globalPunc.replace("'","")
globalPunc = globalPunc.replace("/","")
globalPunc = globalPunc.replace(".","")
globalPunc = globalPunc.replace(";","")
globalPunc = globalPunc.replace("(","")
globalPunc = globalPunc.replace(")","")
key_value = {}
labels, texts,noTitleData, titleData= [], [], [], []
stopwords = ['i','me','my','myself','we','our','ours', 'ourselves', 'you','I','Use','use','Have','CJOC','cjoc','Canada','canada','fit' 'your','yours','yourself','yourselves','he','him','his','himself','she','her','herself','it','its',"it's",'itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','The','A','under','again','further','would','then','once','here','there','when','where','why','how','all','any','-','s',"'s'",'w','',' ','both','each',"aren't",'few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just',"don't",'should','now']
def prepCsv(): #prepares csv file
    trainDF = pd.DataFrame()
    numTrack = 0
    df = pd.read_csv('rimpac1.csv', encoding="cp1252")
    i = (len(df.index))
    print (i)
    while numTrack < i: # iterates through the csv and parses the object
        parseObj(df.iloc[numTrack],numTrack)
        numTrack += 1
        #break
    for key in key_value: # gets the parsed title of each object
        if key_value[key].getImportantWords() is '': # textx with no title doesnt go into our predictions
            print("no words")
        else:
            titleData.append(key_value[key].getImportantWords()) # appends it to a titleData array
            #titleData = (key_value[key].getImportantWords())
        pass
#clustering the csv file based on title data
    print("Title Data: " ,   titleData)
    vect = TfidfVectorizer(stop_words="english")
    xData = vect.fit_transform(titleData)
    # used worrds
    words = vect.get_feature_names()
    print("words: %s" % words)
    n_clusters = 5
    num_of_seeds = 10
    max_iter = 300
    num_of_proc = 2 #seeds distributed
    model = KMeans(n_clusters=n_clusters,max_iter=max_iter,n_init=num_of_seeds,n_jobs=num_of_proc).fit(xData)
    labels = model.labels_
    ordered_words = model.cluster_centers_.argsort()[:, ::-1]
    print("centers:", model.cluster_centers_)
    print("labels:",labels)
    print("intertia:", model.inertia_)
    texts_clust = np.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        for label in labels:
            if label==i_cluster:
                texts_clust[i_cluster] +=1
    print("Top words per cluster")
    for i_cluster in range(n_clusters):
        print("Cluster:",i_cluster,",texts:",int(texts_clust[i_cluster]))
        for term in ordered_words[i_cluster, :3]:
            print("\t"+words[term])
    #making our own csv file
    initCSV()
    #cluster prediction
    for key in key_value:
        text_to_precict = ""
        if key_value[key].getTitle() is '': # textx with no title doesnt go into our predictions
            with open('testingSet.csv',mode='a') as csv_file:
                fieldnames = ['Handle','Title','FullText','Authors','ClusterTerms','CleanedText']
                writer = csv.DictWriter(csv_file,fieldnames=fieldnames,lineterminator = '\n')
                writer.writerow({'Handle':key_value[key].getHandle(), 'Title':key_value[key].getTitle(),'FullText':key_value[key].getText(),'Authors':key_value[key].getAuthors(),'ClusterTerms':'','CleanedText':key_value[key].cleanText()})
                csv_file.close()
        else:
            terms_predicted = []
            stringTerm = ''
            text_to_precict = key_value[key].getTitle()
            yData = vect.transform([text_to_precict])
            predict_cluster = model.predict(yData)[0]
            for term in ordered_words[predict_cluster, :1]:
                stringTerm = words[term]
            with open('trainingSet.csv',mode='a') as csv_file:
                fieldnames = ['Handle','Title','FullText','Authors','ClusterTerms','CleanedText']
                writer = csv.DictWriter(csv_file,fieldnames=fieldnames,lineterminator = '\n')
                writer.writerow({'Handle':key_value[key].getHandle(), 'Title':key_value[key].getTitle(),'FullText':key_value[key].getText(),'Authors':key_value[key].getAuthors(),'ClusterTerms':stringTerm, 'CleanedText':key_value[key].cleanText()})
                csv_file.close()
def classify():
    np.random.seed(5000)
    trainText, trainTerms, stemmedText, = [], [], []
    keepTrack = 0
    df = pd.read_csv('data.csv', encoding="cp1252") #first data set, labled data from bbc
    df['category_id'] = df.category.factorize()[0]
    category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values) #transforming data
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    tfidf_vect = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    faeturesX =  tfidf_vect.fit_transform(df.content).toarray() # transform the text
    labels = df.category_id #labels of the data
    faeturesX.shape
    Train_X,Test_X, Train_Y, Test_Y,indices_train, indices_test= train_test_split(faeturesX,labels,df.index,test_size=0.33,random_state=0)
    texts = ["Staff should arrive at ship with own access in place without using the Ship's allocation.",
             "In order to focus on operator training and reporting recommend units establish voice comms for reporting purposes prior to COMEX rather than relying on chat.",
             "My recommendation is that the liaison between the remoted Tac Avn Det and the ATF is done via an ATF LO embedded within the detachment(SOCAL) The ATF LO in the Tac Avn Det would be the extra manpower required to collate and staff all the reports and returns. He will be intimately familiar with them as he will be from Bagotville. Double hatting this individual as a PAO/Visits Officer would help ensure the ATF LO was fully employed.            The Tac Avn DetCO should visit the ATF early in the deployment to better establish the command relationship and formalise reports and returns processes required. The Tac Avn DetCO stated there is value in having an experienced Tac Avn mission commander as an LO at the ATF who can advise the ATF Comd as and when required particularly with respect Mission Acceptance. The call is made on giving up an experienced Tac Avn mission commander on the understanding he will be somewhat underemployed for periods of time. I would suggest consideration be given to sending an LO to the ATF for the initial setup but once a steady state is achieved recalling him to be better employed at the Tac Avn Det; particularly if lines of communication are secure and robust enough to guarantee the DetCO is always able to function as the primary Tac Avn advisor to the ATF Comd.",
             "Excision releases new album, tops charts in North America!",
             "recommend development LNO trng package ensure full scope LNO duties well understood prior employment LNO As suggestion type training likely good candidate online trng package Alternatively could delivered trng programs courses related CAOC trng"]
    text_features = tfidf_vect.transform(texts)
    kNearestNeighbour(Train_X,Test_X, Train_Y, Test_Y,labels,faeturesX,id_to_category,text_features,texts) # kNearestNeighbour uses same data except for tfidf_gaus
    gausNaive(Train_X,Test_X, Train_Y, Test_Y,labels, faeturesX,id_to_category,text_features,texts) # Gaussaian naive uses the same data
    modelSVM(df,id_to_category,text_features,texts,labels,faeturesX)
    pass
def removeStopwords(text):
    cleanText = []
    for word in text:
        if word.lower() not in stopwords:
            cleanText.append(word)
    return cleanText
def modelSVM(df,id_to_category,text_features,texts,labels,faeturesX): #support vector models
    for index,entry in enumerate(df['content']):
        contentClean = removeStopwords(entry.split())
        df.loc[index,'text_final'] = str(contentClean)
    Train_X,Test_X, Train_Y, Test_Y,indices_train, indices_test= train_test_split(df['text_final'],labels,df.index,test_size=0.33,random_state=0)
    Enconder = LabelEncoder()
    Train_Y = Enconder.fit_transform(Train_Y)
    Test_Y = Enconder.fit_transform(Test_Y)
    tfidf_svm =  TfidfVectorizer(max_features=5000)
    tfidf_svm.fit(df['text_final'])
    Train_X_Tfidf = tfidf_svm.transform(Train_X) #important feature
    Test_X_Tfidf = tfidf_svm.transform(Test_X) #important feature

    SVM = SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    prediction_SVM = SVM.predict(Test_X_Tfidf) #predict the test sample
    CV = 5
    pd.DataFrame(index=range(5))
    entries = []
    accuarcies = cross_val_score(SVM,faeturesX,labels,scoring='accuracy',cv=CV)
    print("==================Support Vector Machines==================")
    for fold_idx,accuracy in enumerate(accuarcies):
        entries.append(("SVM",fold_idx,accuracy))
    cv_df = pd.DataFrame(entries,columns=['Model_name','fold_idx','accuracy'])
    print(cv_df)
    print("Micro : ", f1_score(Test_Y,prediction_SVM,average='micro')*100)
    print("Weighted : ", f1_score(Test_Y,prediction_SVM,average='weighted')*100)
    print("all : ", precision_recall_fscore_support(Test_Y,prediction_SVM,average='weighted'))
    print("Accuracy of SVM: ", accuracy_score(prediction_SVM,Test_Y)*100)     #accuracy_score
    #predicing text
    svmtexts = tfidf_svm.transform(texts)
    predictions = SVM.predict(svmtexts)
    for text, predicted in zip(texts, predictions):
        print('"{}"'.format(text))
        print(" - Predicted as: '{}'".format(id_to_category[predicted]))
        print("")
    #Train_X_Tfidf_gaus = tfidf_vect.fit_transform(Train_X)
def gausNaive(Train_X,Test_X, Train_Y, Test_Y,labels,faeturesX,id_to_category,text_features,texts): # better accuracy but not used for predicting text classification
    #gaussian classifer
    gausModel = GaussianNB()
    gausModel.fit(Train_X,Train_Y)
    print()
    print("==================Gaus Naive Bayes==================")
    CV = 5
    entriesGaus = []
    accuarcieGaus = cross_val_score(gausModel,faeturesX,labels,scoring='accuracy',cv=CV)
    for fold_idxGaus,accuracyGaus in enumerate(accuarcieGaus):
        entriesGaus.append(("Gaus",fold_idxGaus,accuracyGaus))
    cv_dfGaus = pd.DataFrame(entriesGaus,columns=['Model_name','fold_idx','accuracy'])
    print(cv_dfGaus)

    prediction_Gaus = gausModel.predict(Test_X)
    print("Micro : ", f1_score(Test_Y,prediction_Gaus,average='micro')*100)
    print("Weighted : ", f1_score(Test_Y,prediction_Gaus,average='weighted')*100)
    print("all : ", precision_recall_fscore_support(Test_Y,prediction_Gaus,average='weighted'))
    print("Accuracy of Gaus: ", accuracy_score(prediction_Gaus,Test_Y)*100)
    #predicition
    predictions = gausModel.predict(text_features.todense())
    for text, predicted in zip(texts, predictions):
        print('"{}"'.format(text))
        print(" - Predicted as: '{}'".format(id_to_category[predicted]))
        print("")
    pass
def kNearestNeighbour(Train_X,Test_X, Train_Y, Test_Y,labels,faeturesX,id_to_category,text_features,texts): #kNearestNeighbour Function
    kNNModel = KNeighborsClassifier(n_neighbors=5)
    kNNModel.fit(Train_X,Train_Y)
    prediction_kNN = kNNModel.predict(Test_X)
    print()
    print("==================kNearestNeighbour==================")
    CV = 5
    entrieskNN = []
    accuarciekNN = cross_val_score(kNNModel,faeturesX,labels,scoring='accuracy',cv=CV)
    for fold_idxkNN,accuracykNN in enumerate(accuarciekNN):
        entrieskNN.append(("KNN",fold_idxkNN,accuracykNN))
    cv_dfkNN = pd.DataFrame(entrieskNN,columns=['Model_name','fold_idx','accuracy'])
    print(cv_dfkNN)

    print("Micro : ", f1_score(Test_Y,prediction_kNN,average='micro')*100)
    print("Weighted : ", f1_score(Test_Y,prediction_kNN,average='weighted')*100)
    print(2*((0.9364392419289688 * 0.9348314606741573) / (0.9364392419289688 + 0.9348314606741573))) # actual f_Score
    print("all : ", precision_recall_fscore_support(Test_Y,prediction_kNN,average='weighted'))
    print("Accuracy of KNN: ", accuracy_score(Test_Y,prediction_kNN)*100)
    predictions = kNNModel.predict(text_features.todense())
    for text, predicted in zip(texts, predictions):
        print('"{}"'.format(text))
        print(" - Predicted as: '{}'".format(id_to_category[predicted]))
        print("")
    pass
def logRegression(): # log regression works, worked on the other data set, but rimapc is too small of a dataset so it needs more entrise for it to go through regression
    ''' labels = terms, category = text '''
    keepTrack = 0
    trainText = []
    trainTerms = []
    df1 = pd.read_csv('trainingSet.csv', encoding="cp1252")
    df1['terms_id'] = df1['ClusterTerms'].factorize()[0]
    print(df1['terms_id'])
    terms_id_df = df1[['ClusterTerms','terms_id']].drop_duplicates().sort_values('terms_id')
    terms_to_id = dict(terms_id_df.values)
    id_to_category = dict(terms_id_df[['terms_id', 'ClusterTerms']].values)
    print(terms_id_df)
    print(terms_to_id)
    print(id_to_category)
    tfidfReg = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidfReg.fit_transform(df1['FullText']).toarray()
    labels = df1.terms_id
    print(features)
    features.shape
    N = 3
    for ClusterTerms, terms_id in sorted(terms_to_id.items()):
        features_chi2 = chi2(features, labels == terms_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidfReg.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    model = LogisticRegression(random_state=0)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df1.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)

def initCSV(): #making a new csv file
    with open('trainingSet.csv',mode='w') as csv_file: # opens file
        fieldnames = ['Handle','Title','FullText','Authors','ClusterTerms','CleanedText']
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames,lineterminator = '\n')
        writer.writeheader()
        csv_file.close()
    with open('testingSet.csv',mode='w') as csv_file: # opens file
        fieldnames = ['Handle','Title','FullText','Authors','ClusterTerms','CleanedText']
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames,lineterminator = '\n')
        writer.writeheader()
        csv_file.close()
def parseObj(obj,numid): # parse the csv file into an obj
    key_value[numid] = csvObj()
    handle = obj['Handle']
    title = obj['Title']
    fullText = obj['FullText']
    authors = obj['Authors']
    fullText = cleanText(fullText)
    key_value[numid].setHandle(handle,fullText,authors)
    key_value[numid].parseTitle(title)
    key_value[numid].calcWeight()

def cleanText(fullText): # cleaning text function
    try:
        returnText = fullText
        returnText = returnText.replace('\n','')
        returnText = returnText.replace('•','')
        returnText = returnText.replace(u'\xa0',u' ')
        returnText = re.sub('[%s]' % re.escape(globalPunc),'',returnText)
        return returnText
    except AttributeError:
        print("Full: %s" % fullText)
        print("nan")
        return "empty"
#prepCsv()
classify()
