import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from config import config
nJoints = config.NETWORK.nJoints

class Residual(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.fc1  = nn.Dense(1024)
        self.bn1  = nn.BatchNorm()
        self.act1 = nn.Activation('relu')
        self.dp1  = nn.Dropout(0.5)
        self.fc2  = nn.Dense(1024)
        self.bn2  = nn.BatchNorm()
        self.act2 = nn.Activation('relu')
        self.dp2  = nn.Dropout(0.5)
        
    def hybrid_forward(self,F,x):
        out = self.dp1(F.relu(self.bn1(self.fc1(x))))
        out = self.dp2(F.relu(self.bn2(self.fc2(out))))
        return out + x

class SimpleBaseline(nn.HybridBlock):
    def __init__(self, cfg, verbose=False, **kwargs):
        super(SimpleBaseline, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            self.net= nn.HybridSequential()
            self.net.add(
                nn.Dense(1024),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Dropout(0.5),
                Residual(),
                Residual(),
                nn.Dense(3*nJoints))

    def hybrid_forward(self,F,x):
        out = x
        for i,b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

def get_net(cfg):
    return SimpleBaseline(cfg)