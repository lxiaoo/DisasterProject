import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    1 读取文件&合并
    输入参数：
        messages_filepath: string, 消息文件地址
        categories_filepath: string, 分类文件地址
    输出参数：
        df:包含消息和分类文件信息的dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='inner', on=['id'])
    return df
    
    
def clean_data(df):
    """
    #2 转换categories文件格式
    输入参数：
        df: dataframe，包含消息和分类文件信息的dataframe
    输出参数：
        df: 对输入参数进行转换，包括转换标题，转换值，去重
    """
    categories = df['categories']
    
    #2.1 转换标题
    categories_id = df['id']
    categories = categories.str.split(";",expand=True)
    row = categories.values[0]
    category_colnames = pd.Series(row).apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    #2.2 转换值
    for column in categories.columns:
        categories[column] = categories[column].str[-1]
    categories[column] = categories[column].astype(int)
    categories[column] = categories[column].apply(lambda x: 0 if x==0 else 1 )
    #2.3 合并ID
    categories = pd.concat([categories_id,categories],axis=1)
    
    #3 合并message和categories,去重
    df.drop(['categories'],axis=1,inplace=True)
    df = df.merge(categories,how='inner',on=['id'])
    df = df.drop_duplicates(subset=['id'])
    
    return df

def save_data(df, database_filename):
    """
    #4 输出
    输入参数：
        df: dataframe，清洗过的dataframe
        database_filename: string, 数据库文件地址
    输出参数：
        X: 特征参数
        y: 分类参数
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessagesCat', con = engine, if_exists='replace', index=False)
    
    #5 define features and label arrays
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    
    return X,y  


def main():
    """
    执行函数
        无输入参数
        无输出参数
    执行过程：
        1. 导入源数据
        2. 清洗数据
        3. 保存数据到数据库    
    
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()