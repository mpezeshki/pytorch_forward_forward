def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_



class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        # 定义一个包含n个数据的列表
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def avg(self):
        return [sum(a)/len(a) for a in self.data]

    def __getitem__(self, idx):
        return self.data[idx]



def accuracy(y_hat, y):  # @save

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:

        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())