import torch, numpy as np, matplotlib.pyplot as plt, torch, time
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from adabound import AdaBound
import dataset

class ResNetClassifier(nn.Module):
    def __init__(self, pretrained=(True, True), resnet_depth=34):
        super(ResNetClassifier, self).__init__()
        self.__resnet_depth = resnet_depth
        self.__pretrained, self.__continualTraining = pretrained

        if self.__resnet_depth == 18:
            self.resnet = resnet18(pretrained=self.__pretrained)
        elif self.__resnet_depth == 101:
            self.resnet = resnet101(pretrained=self.__pretrained)
        elif self.__resnet_depth == 50:
            self.resnet = resnet50(pretrained=self.__pretrained)
        elif self.__resnet_depth == 152:
            self.resnet = resnet152(pretrained=self.__pretrained)
        else:
            self.resnet = resnet34(pretrained=self.__pretrained)
        
        if self.__continualTraining:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.linear_layer = nn.Sequential(
            nn.Linear(1000, 256, bias=True),
            nn.Linear(256, 64, bias=True),
            nn.Linear(64, 16, bias=True),
            nn.Linear(16, 1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.linear_layer(x)
        return x

class NNProcess:
    def __init__(self, 
            import_trained=(False, ''),
            model_pretrained=(True, True),
            save_model=True,
            resnet_depth=50,
            lr=1e-3,
            momentum=0.09,
            nesterov=False,
            threshold=0.5,
            epochs=50,
            batch_size=64,
            train_val_split=0.7,
            data_interval='1min',
            predict_period=1,
            mins_interval=30,
            start_date='2020-08-24',
            end_date='2020-08-29'
        ):

        '''
        import_trained = (whether if you want to import a trained pth file, if yes what is the filename)
        model_pretrained = (whether if you want to import a pretrained model, whether if you want to only want to train the linear layers)
        save_model = whether to save model when training finished
        resnet_depth = to decide the depth of the residual network
        lr = learning rate for the stochastic gradient descend optimizer
        momentum = momentum for the sgd
        nesterov = whether to use nesterov momentum for sgd
        threshold = investment threshold, advices to invest if the returned probability > threshold
        epochs = training hyperparameter: the number of times the entire dataset is exposed to the neural network
        batch_size = training hyperparameter: the number of items to show the dataset at once
        train_val_split = training hyperparameter: how to split the data
        data_interval = the time interval between each datapoint
        predict_period = the amount of time period to predict forwards
        days = the amount of days to use
        mins_interval = the amount of minutes to show in the graph
        start_date = the first date to get data - data for each day would start from 9am and end at 8pm
        end_date = the last date to get data - data for each day would start from 9am and end at 8pm
        '''

        self.__import_trained = import_trained
        self.__model_pretrained = model_pretrained
        self.__saveModel = save_model
        self.__resnet_depth = resnet_depth
        self.__threshold = threshold
        self.__epochs = epochs
        self.__batch_size = batch_size
        data = dataset.stockGraphGenerator(split=train_val_split, interval=data_interval, predict_period=predict_period, mins_interval=mins_interval, start_date=start_date, end_date=end_date, stride=15)
        self.__train_set = torch.utils.data.DataLoader(data.train_data, batch_size=self.__batch_size, shuffle=False)
        self.__test_set = torch.utils.data.DataLoader(data.test_data, batch_size=self.__batch_size, shuffle=False)
        self.__model = self.__loadmodelInstance() if self.__import_trained[0] else self.__createmodelInstance()
        self.__criterion = nn.BCELoss()
        self.__optim = AdaBound(self.__model.parameters(), amsbound=True, lr=lr, final_lr=0.1)
        self.__trainHist = [[], [], [], []]

    def __loadmodelInstance(self):
        model = torch.load(self.__import_trained[1]+'.pth')
        return model.cuda() if torch.cuda.is_available() else model

    def __createmodelInstance(self):
        return ResNetClassifier(pretrained=self.__model_pretrained, resnet_depth=self.__resnet_depth).cuda() if torch.cuda.is_available() else ResNetClassifier(pretrained=self.__model_pretrained, resnet_depth=self.__resnet_depth)

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def __setModelTrain(self):
        self.__model = self.__model.train()

    def __setModelEval(self):
        self.__model = self.__model.eval()

    def fetch_model(self):
        return self.__model.eval()

    def fetch_trainHist(self):
        return self.__trainHist

    def __get_acc(self, output, label):
        output = torch.round(output)
        num_correct = sum([1 if output[i]==label[i] else 0 for i in range(len(output))])
        return num_correct / output.shape[0]

    def train(self):
        for epochs in range(self.__epochs):
            start_time = time.time()
            avg_train_loss, avg_test_loss = 0, 0
            avg_train_acc, avg_test_acc = 0, 0
            train_total, test_total = 0, 0
            self.__setModelTrain()
            for im, label in self.__train_set:
                train_total += 1
                im, label = Variable(im), Variable(label)
                pred = self.__model(im)
                train_loss = self.__criterion(pred, label)
                self.__optim.zero_grad()
                train_loss.backward()
                self.__optim.step()
                avg_train_loss += train_loss.data.tolist()
                avg_train_acc += self.__get_acc(pred, label)
                print ('Training Batch No.: {:3d}\nTraining Loss: {:.5f} ; Training Acc.: {:.5f}'.format(train_total, train_loss.data.tolist(), self.__get_acc(pred, label)))

            self.__setModelEval()
            for im, label in self.__test_set:
                test_total += 1
                im, label = Variable(im, requires_grad=False), Variable(label, requires_grad=False)
                pred = self.__model(im)
                test_loss = self.__criterion(pred, label)
                avg_test_loss += test_loss.data.tolist()
                avg_test_acc += self.__get_acc(pred, label)
                print ('Testing Batch No.: {:3d}\nTesting Loss: {:.5f} ; Testing Acc.: {:.5f}'.format(test_total, test_loss.data.tolist(), self.__get_acc(pred, label)))

            self.__trainHist[0].append(avg_train_loss/train_total)
            self.__trainHist[1].append(avg_test_loss/test_total)
            self.__trainHist[2].append(avg_train_acc/train_total)
            self.__trainHist[3].append(avg_test_acc/test_total)
            print ('Epoch: {:3d} / {:3d}\nAverage Training Loss: {:.6f} ; Average Validation Loss: {:.6f}\nTrain Accuracy: {:.3f} ; Test Accuracy: {:.3f}\nTime Taken: {:.6f}\n'.format(epochs+1, self.__epochs, avg_train_loss/train_total, avg_test_loss/test_total, avg_train_acc/train_total, avg_test_acc/train_total, time.time()-start_time))


        if self.__saveModel:
            torch.save(self.__model, './resnet_market_predictor.pth')
