import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessagesCat',con=engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:].astype('int')
    col_names = y.columns
    return X, y, col_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #去掉标点
    tokens = word_tokenize(text) #分词
    lemmatizer = WordNetLemmatizer() #提取词干
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))  #SVC(), RandomForestClassifier(), MultinomialNB() #
                                                #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    ])
    parameters = {
    'moc__estimator__criterion': ['gini','entropy'],
    'moc__estimator__n_estimators': [10,15]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    scores = pd.DataFrame(data=None, index=category_names, columns =['accuracy','precision','recall','F1','True_cnt','False_cnt'],dtype='float')
    test_cnt = Y_test.shape[0]
    for i in range(y_pred.shape[1]):
        #print(i)
        col_pred = y_pred[:,i]
        col_true = np.array(Y_test,ndmin=2)[:,i]
        #print(classification_report(col_true,col_pred))
        scores['accuracy'].iloc[i] = accuracy_score(col_true,col_pred)
        scores['precision'].iloc[i] = precision_score(col_true, col_pred,average='micro')
        scores['recall'].iloc[i] = recall_score(col_true, col_pred,average='micro')
        scores['F1'].iloc[i] = f1_score(col_true, col_pred,average='micro')
        scores['True_cnt'].iloc[i] = np.sum(col_true)
        scores['False_cnt'].iloc[i] = test_cnt-np.sum(col_true)
    return scores


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()