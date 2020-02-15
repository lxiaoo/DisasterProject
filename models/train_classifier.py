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
    '''
    # 从数据读取数据，包括标签和消息
    输入函数：
        database_filepath: string, 数据库文件地址
    输出函数：
        X: 特征
        y: 分类
        col_names: 分类的列名
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessagesCat',con=engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:].astype('int')
    col_names = y.columns
    return X, y, col_names

def tokenize(text):
    """
    #处理并标准化每个词汇
    输入函数：
        text: string，需要处理的字符串
    输出函数：
        clean_tokens：清洗后的字符串列表，包括去掉标点、分词、提取词干等处理
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #去掉标点
    tokens = word_tokenize(text) #分词
    lemmatizer = WordNetLemmatizer() #提取词干
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    #建立模型，并选择预测效果最好的模型
    输入函数：没有
    输出函数：
        cv: 预测模型
    """
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
    """
    #把模型预测结果的分数保存到data frame
    输入参数：
        model: 分类模型
        X_test: dataframe, 测试特征参数
        Y_test: dataframe, 测试预测结果
        category_names: list, 分类的列名
    输出参数：
        scores: 各个分类的预测结果评分
    """
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
    """
     #保存模型到pickle文件
     输入参数：
        model: 分类模型
        model_filepath: 输出模型的文件地址
     输出参数：无
    """
    
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)



def main():
    """
    主程序：
    输入函数：无
    输出函数：无
    处理过程：
        1. 从数据库加载数据到X,y
        2. 根据X和y的数据建立模型
        3. 评估模型效果
        4. 保存模型到pickle文件
    """
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