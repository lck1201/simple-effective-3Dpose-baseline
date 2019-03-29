import time

from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd

from lib.utils import saveModel

def trainNet(net, trainer, train_data, loss, train_metric, epoch, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    train_data.reset()  # reset and re-shuffle
    if train_metric:
        train_metric.reset()

    trainloss, n = [0] * len(ctx), 0
    RecordTime = {'load': 0, 'forward': 0, 'loss': 0, 'backward': 0, 'post': 0}

    for batch_i, batch in enumerate(train_data):
        beginT = time.time()
        data_list = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label_list = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        RecordTime['load'] += time.time() - beginT

        beginT = time.time()
        Ls = []
        output_list = []
        with autograd.record():
            for x, y in zip(data_list, label_list):
                preds = net(x)
                L = loss(preds, y)
                Ls.append(L)
                output_list.append(preds)
            RecordTime['forward'] += time.time() - beginT

            beginT = time.time()
            for L in Ls:
                L.backward()
            RecordTime['loss'] += time.time() - beginT

        beginT = time.time()
        trainer.step(batch.data[0].shape[0])
        RecordTime['backward'] += time.time() - beginT

        beginT = time.time()
        # Number
        n += batch.data[0].shape[0]

        # Loss
        for i in range(len(trainloss)):
            trainloss[i] += Ls[i]

        # MPJPE
        if config.TRAIN.UseMetric:
            for lb, pd in zip(label_list, output_list):
                train_metric.update(lb, pd)
        RecordTime['post'] += time.time() - beginT

    totalT = nd.array([RecordTime[k] for k in RecordTime]).sum().asscalar()
    for key in RecordTime:
        print("%-s: %.1fs %.1f%% " % (key, RecordTime[key], RecordTime[key] / totalT * 100), end=" ")
    print(" ")

    nd.waitall()
    trainloss = sum([item.sum().asscalar() for item in trainloss])
    MPJPE = train_metric.get()[-1].sum(axis=0) / 17 if config.TRAIN.UseMetric else 0
    logger.info("TRAIN - Epoch:%d LR:%.2e Loss:%.2e MPJPE:%.1f" % (epoch + 1, trainer.learning_rate, trainloss / n, MPJPE))

    # save model
    if ((epoch + 1) % (config.TRAIN.end_epoch / 5) == 0 or epoch == 0):
        saveModel(net, logger, config, isCKP=True, epoch=epoch + 1)
    if (epoch + 1 == config.TRAIN.end_epoch):
        saveModel(net, logger, config, isCKP=False, epoch=epoch + 1)


def validNet(net, valid_data, loss, eval_metric, epoch, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    valid_data.reset()
    if eval_metric:
        eval_metric.reset()

    validloss, n = [0] * len(ctx), 0
    for batch_i, batch in enumerate(valid_data):
        data_list = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label_list = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

        Ls = []
        output_list = []
        for x, y in zip(data_list, label_list):
            preds = net(x)
            L = loss(preds, y)
            output_list.append(preds)
            Ls.append(L)

        if config.TRAIN.UseMetric:
            for lb, pd in zip(label_list, output_list):
                eval_metric.update(lb, pd)

        for i in range(len(validloss)):
            validloss[i] += Ls[i]
        n += batch.data[0].shape[0]

    nd.waitall()
    validloss = sum([item.sum().asscalar() for item in validloss])
    MPJPE = eval_metric.get()[-1].sum(axis=0) / 17
    logger.info("VALID - Epoch:%d Loss:%.3e MPJPE:%.1f" % (epoch + 1, validloss / n, MPJPE))


def TestNet(net, test_data, loss, avg_metric, xyz_metric, mean3d, std3d, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    test_data.reset()
    if avg_metric:
        avg_metric.reset()

    # nJoints = config.NETWORK.nJoints
    TestLoss, n = [0] * len(ctx), 0

    begintime = time.time()
    for batch_i, batch in enumerate(test_data):
        # get data
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        folders = gluon.utils.split_and_load(batch.label[1], ctx_list=ctx, batch_axis=0)

        # forward
        Ls = []
        output_list = []
        for x, y in zip(data, label):
            preds = net(x)
            L = loss(preds, y)
            output_list.append(preds)
            Ls.append(L)

        # update loss&metric
        for label_batch, pred_batch in zip(label, output_list):
            avg_metric.update(label_batch, pred_batch)
            xyz_metric.update(label_batch, pred_batch)

        for i in range(len(TestLoss)):
            TestLoss[i] += Ls[i]

        n += batch.data[0].shape[0]

    endtime = time.time()

    # calc error
    MPJPE = avg_metric.get()[-1].sum(axis=0) / 17
    jntErr = avg_metric.get()[-1]
    xyzErr = xyz_metric.get()[-1]

    DBsize = n
    TestLoss = sum([item.sum().asscalar() for item in TestLoss])
    Losses = TestLoss / n

    Time = endtime - begintime

    return [DBsize, Losses, Time, MPJPE, xyzErr, jntErr]
