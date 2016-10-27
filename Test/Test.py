# -*- coding:utf-8 -*-

import numpy as np


class Test():
    def __init__(self):
        self.a = np.random.rand(2, 1)

    def run(self):
        b = [7, 8, 9]

        print self.a[1]

        ####################
        c = zip(self.a, b)
        for i, j in c:
            i += 1

        print self.a[1]

        ####################
        for i, j in zip(self.a, b):
            i += 10

        print self.a[1]

        ####################
        for i in self.a:
            i += 100

        print self.a[1]

        # 采用Adagrad自适应梯度下降法,可参看博文：http://blog.csdn.net/danieljianfeng/article/details/42931721
        # for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
        #                               [dWxh, dWhh, dWhy, dbh, dby],
        #                               [mWxh, mWhh, mWhy, mbh, mby]):
        #     if n == 2:
        #         print len(param), len(dparam), len(mem)
        #
        #     mem += dparam * dparam
        #     param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # 自适应梯度下降公式

        #
        # for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby],
        #                               [mWxh, mWhh, mWhy, mbh, mby]):
        #     mem += dparam * dparam
        #     param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # 自适应梯度下降公式


if __name__ == "__main__":
    test = Test()
    test.run()
