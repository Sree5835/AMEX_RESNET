import torch, datetime, gc, matplotlib.pyplot as plt
import utils

gc.enable()

class stockGraphGenerator(torch.utils.data.Dataset):
    def __init__(self, split=0.7, interval='1min', predict_period=1, days=5, mins_interval=30, start_date='2020-08-24', end_date='2020-08-29', stock_name='SPY', stride=1):
        super(stockGraphGenerator, self).__init__()
        self.__start_date = datetime.datetime.strptime(start_date+' 10:00:00', '%Y-%m-%d %H:%M:%S')
        self.__end_date = datetime.datetime.strptime(end_date+' 20:00:00', '%Y-%m-%d %H:%M:%S')
        self.__mins_interval = mins_interval
        self.__stride = stride
        self.__data_len = self.__calculateLen(days, mins_interval)
        self.__interval = interval
        self.__predict_period = predict_period
        self.__data_raw = utils.getData(stock_name).reset_index()
        self.train_data = torch.utils.data.Subset(self, list(range(0, int(split*self.__data_len))))
        self.test_data = torch.utils.data.Subset(self, list(range(int(split*self.__data_len), int(self.__data_len))))

    def __calculateLen(self, days, mins_interval):
        return (((((20-10)*60)-mins_interval)/self.__stride)+1)*days

    def __getitem__(self, x):
        #everyday for this program starts from 10am to 8pm
        selected_date = self.__start_date + datetime.timedelta(minutes=self.__stride*x)
        return utils.stock_pic(self.__data_raw, self.__interval, selected_date, selected_date + datetime.timedelta(minutes=self.__mins_interval), self.__predict_period)

    def __len__(self):
        return self.__data_len

'''
data = stockGraphGenerator()
train_data = data.train_data
test_data = data.test_data
train_set = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
test_set = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

for i in train_set:
    print (i.shape)
'''
