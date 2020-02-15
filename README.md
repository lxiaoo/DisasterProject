# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

###项目总结###
目的：在灾害发生时，根据人们在互联网上输入的各种信息，判断他们需要什么样的帮助
实现方式：
1. 建模
根据已有的灾害发生时的消息和打好的标签，建立模型预测未来的消息和判断求助者可能想表达的信息
1.1 清洗已有的灾害发生时的消息和打好的标签，把结果输入数据库
1.2 根据清洗完的数据，建立预测模型，输出为pickle文件，供前端页面调用
2. 网页
2.1 通过已建立的模型，当用户在页面输入消息时判断可能需要的信息的标签
2.2 通过已有的消息和标签，分别展示消息来源，目前消息请求最多的灾害，和不同灾害需要的协助请求多少




