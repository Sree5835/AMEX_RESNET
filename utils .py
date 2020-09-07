import pandas as pd, numpy as np, os, torch, gc, datetime
import matplotlib.dates as mdates, matplotlib.pyplot as plt, matplotlib as mpl
import torchvision.transforms.functional as TF
from pandas_datareader import data
from alpha_vantage.timeseries import TimeSeries
from PIL import Image

gc.enable()

def key():
    API_key = pd.read_csv('API_key')
    return (API_key['API Key'][np.random.randint(0,len(API_key),1)[0]])

def __ImageToTensor(fig):
    fig.savefig('./fig.png')
    tensor = TF.to_tensor(Image.open('./fig.png').convert('RGB')).unsqueeze_(0)[0]
    os.remove('./fig.png')
    return tensor
'''
def getData(ticker, interval='1min'):
    time=TimeSeries(key=key(),output_format='pandas')
    data=time.get_intraday(symbol=ticker,interval=interval,outputsize='full')
    return data[0]
'''

def getData(ticker, interval='1min'):
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

def stock_pic(data,time_interval,start_date,end_date,gradient_period):
    x = data[data['date']==str(end_date)].index.values
    y = data[data['date']==str(start_date)].index.values
    if not x:
        x = x + 1 
    elif not y:
        y = y + 1 
    else:
        pass
    
    data['6. AVG'] = (data['1. open'] + data['4. close'])/2
    data['EWMA-AVG'] = data['6. AVG'].ewm(span=3).mean()
    gradient = (data['EWMA-AVG'][int(x)]-data['EWMA-AVG'][int(x)-gradient_period])/gradient_period
    grad_label = torch.ones(1) if gradient > 0 else torch.zeros(1)
    
    data = data[int(x):int(y)]
    data = data.drop(['5. volume'], axis=1)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
    ax.plot(data['1. open'],c='g',alpha=1)
    ax.plot(data['4. close'],c='r',alpha=1)
    ax.plot(data['EWMA-AVG'],c='b',alpha=1)
    plt.axis('off')
    plt.show()

    return __ImageToTensor(fig).cuda() if torch.cuda.is_available() else __ImageToTensor(fig), grad_label.cuda() if torch.cuda.is_available() else grad_label
