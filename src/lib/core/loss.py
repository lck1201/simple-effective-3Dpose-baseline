from mxnet.gluon.loss import Loss
from mxnet.gluon.loss import _reshape_like
from mxnet.gluon.loss import _apply_weighting

class MeanSquareLoss(Loss):
    r"""Calculates the mean squared error between `pred` and `label`.
        source code in tf: self.loss = tf.reduce_mean(tf.square(y - dec_out))
    """
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(MeanSquareLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
