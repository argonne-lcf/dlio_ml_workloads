import horovod.tensorflow.keras as hvd
hvd.init()
print(hvd.size(), hvd.rank())
