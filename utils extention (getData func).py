def getData():
    dataset = pd.read_csv('FUTURES MINUTE.txt', header = None)
    dataset.columns = ['Date','time',"1. open","2. high",'3. low','4. close','5. volume']
    dataset['date'] = dataset['Date'] +" "+ dataset['time']
    
    dataset.drop('Date', axis=1, inplace=True)
    dataset.drop('time', axis=1, inplace=True)
    dataset['date'] = dataset['date'].apply(lambda x: pd.to_datetime(x, errors='ignore'))
    dataset['date'] = dataset['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    dataset.set_index(dataset.index.map(lambda x: pd.to_datetime(x, errors='ignore')))
    dataset.set_index('date',inplace=True)

    return dataset
