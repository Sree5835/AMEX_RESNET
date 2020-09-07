import nnModels
import matplotlib.pyplot as plt, warnings

warnings.filterwarnings("ignore")

model = nnModels.NNProcess(
            import_trained=(False, ''),
            model_pretrained=(True, True),
            save_model=True,
            resnet_depth=34,
            lr=1e-3,
            momentum=0.09,
            nesterov=False,
            threshold=0.5,
            epochs=50,
            batch_size=32,
            train_val_split=0.7,
            data_interval='1min',
            predict_period=1,
            days=5,
            mins_interval=30,
            start_date='2020-08-24',
            end_date='2020-08-29'
        )

model.train()

history = model.fetch_trainHist()
architecture = model.fetch_model()

plt.plot(history[0])
plt.plot(history[1])
plt.plot(history[2])
plt.plot(history[3])
plt.show()
