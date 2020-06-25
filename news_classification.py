import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from tqdm import tqdm
np.random.seed(500)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(ngram_range=(1,3))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')


df = pd.read_json('news_data.json', lines=True)
print(df.info())
df.category[(df['category']=='ARTS') | (df['category']=='CULTURE & ARTS')]='ARTS & CULTURE'
df.category[df['category']=='PARENTS']='PARENTING'
df.category[df['category']=='STYLE']='STYLE & BEAUTY'
df.category[df['category']=='THE WORLDPOST']='WORLDPOST'
labels = list(df.category.unique())
labels.sort()
print(labels)

def preprocessing(col,h_pct,l_pct):
    #Lower case
    lower = col.apply(str.lower)
    
    #Removing HTML tags
    rem_html = lower.apply(lambda x: x.replace('#39;', "'").replace('amp;', '&')
                             .replace('#146;', "'").replace('nbsp;', ' ').replace('#36;', '$')
                             .replace('\\n', "\n").replace('quot;', "'").replace('<br />', " ")
                             .replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.')
                             .replace(' @-@ ','-').replace('\\', ' \\ ').replace('&lt;','<')
                             .replace('&gt;', '>'))
    
    #Stemming
    stem = SnowballStemmer('english')
    stemmed = rem_html.apply(lambda x: ' '.join(stem.stem(word) for word in str(x).split()))
    
    #removing punctuation
    import re
    rem_punc = stemmed.apply(lambda x: re.sub(r'[^\w\s]',' ',x))
    
    #removing stopwords and extra spaces
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    rem_stopwords = rem_punc.apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    
    #removing numbers
    rem_num = rem_stopwords.apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
    
    #remove words having length=1
    rem_lngth1 = rem_num.apply(lambda x: re.sub(r'[^\w\s]',' ',x))
    
    if h_pct != 0:
        #removing the top $h_pct of the most frequent words 
        high_freq = pd.Series(' '.join(rem_lngth1).split()).value_counts()[:int(pd.Series(' '.join(rem_lngth1).split()).count()*h_pct/100)]
        rem_high = rem_lngth1.apply(lambda x: " ".join(x for x in x.split() if x not in high_freq))
    else:
        rem_high = rem_lngth1
    
    if l_pct != 0:
        #removing the top $l_pct of the least frequent words
        low_freq = pd.Series(' '.join(rem_high).split()).value_counts()[:-int(pd.Series(' '.join(rem_high).split()).count()*l_pct/100):-1]
        rem_low = rem_high.apply(lambda x: " ".join(x for x in x.split() if x not in low_freq))
    else:
        rem_low = rem_high
    
    return rem_low
	
h_pct=0
l_pct=1
verbose = True
df['short_description_processed'] = preprocessing(df['short_description'],h_pct,l_pct)
df['concatenated'] = df['headline'] + '\n' + df['short_description_processed']
    #not removing high and low frequency words from headline
    #this is because the headline carries more significance in determining the classification of the news
df['concat_processed'] = preprocessing(df['concatenated'],0,0)
    
if verbose:
        print('Number of words in corpus before processing: {}'
              .format(df['short_description'].apply(lambda x: len(x.split(' '))).sum()))
        print('Number of words in corpus after processing: {} ({}%)'
              .format(df['short_description_processed'].apply(lambda x: len(x.split(' '))).sum()
                     , round(df['short_description_processed'].apply(lambda x: len(x.split(' '))).sum()*100\
                             /df['short_description'].apply(lambda x: len(x.split(' '))).sum())))
        print('Number of words in final corpus: {} ({}%)'
              .format(df['concat_processed'].apply(lambda x: len(x.split(' '))).sum()
                     , round(df['concat_processed'].apply(lambda x: len(x.split(' '))).sum()*100\
                             /df['short_description'].apply(lambda x: len(x.split(' '))).sum())))
        print('\nRaw story:\n{}'.format(df['short_description'][58142]))
        print('\nProcessed story:\n{}'.format(df['short_description_processed'][58142]))
        print('\nAdding additional columns to story:\n{}'.format(df['concatenated'][58142]))
        print('\nFinal story:\n{}'.format(df['concat_processed'][58142]))
X = df['concat_processed']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=False) 
        
bow_xtrain = bow.fit_transform(X_train)
bow_xtest = bow.transform(X_test)
print(bow_xtrain)
print(bow_xtest)

if verbose:
			print('\nPredicted class: {}'.format(preds[58142]))
            print('Actual class: {}\n'.format(y_test.iloc[58142]))
            plt.figure(figsize=(14,14))
            sns.heatmap(confusion_matrix(y_test,preds),cbar=False,annot=True,square=True,xticklabels=labels,yticklabels=labels,fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.yticks(rotation=0)
            plt.show()
            print(classification_report(y_test,preds))
            print('Accuracy: {0:.2f}%'.format(acc))
			
model = LinearSVC()
model.fit(bow_xtrain,y_train)
preds = model.predict(bow_xtest)
acc = accuracy_score(y_test,preds)*100
	
