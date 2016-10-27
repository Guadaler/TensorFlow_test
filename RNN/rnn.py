#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------
# 功能：基于最简单BPTT的RNN
# Author: goodluck
# Software: PyCharm Community Edition
# File: rnn.py
# Time: 16-10-27 下午3:05
# Url: 代码参考 http://blog.csdn.net/hjimce/article/details/49095371
# -------------------------------------------

import numpy as np


class Rnn:
    def __init__(self):
        self.data = open('text_3', 'r').read().decode('utf-8')  # 读取txt一整个文件的内容为字符串str类型

        print '[Data长度] ', len(self.data)
        print ','.join(self.data)

        self.chars = list(set(self.data))  # 去除重复的字符
        print ','.join(self.chars)

        # 打印源文件中包含的字符个数、去重后字符个数
        self.data_size, self.vocab_size = len(self.data), len(self.chars)
        print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size)

        # 创建字符的索引表
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        # print "#".join(self.char_to_ix)
        for i in self.char_to_ix.keys():
            print i, ':', self.char_to_ix[i]

        self.hidden_size = 100  # 隐藏层神经元个数
        self.seq_length = 20  # 序列长度？？？ 或者是输出层神经元的个数？？窗口大小
        self.learning_rate = 1e-1  # 学习率

        # 网络模型
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01  # U 输入层到隐藏层, 标准随机正态分布矩阵
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01  # W 隐藏层与隐藏层
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01  # V 隐藏层到输出层，输出层预测的是每个字符的概率
        self.bh = np.zeros((self.hidden_size, 1))  # 隐藏层偏置项
        self.by = np.zeros((self.vocab_size, 1))  # 输出层偏置项

    # inputs  t时刻序列，也就是相当于输入
    # targets t+1时刻序列，也就是相当于输出
    # hprev t-1时刻的隐藏层神经元激活值
    # 以上三个参数：t时刻隐藏层的两个输入（inputs，hprev）+ 一个输出（targets）， targets表示真实输出
    def lossFun(self, inputs, targets, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        # 前向传导, 下标t表示t时刻
        for t in xrange(len(inputs)):  # range()和xrange()的区别
            # 构建输入向量，用one hot 向量表示输入的那个字，如 "我" ：[1, 0, 0, 0, 0, 0, 0]
            xs[t] = np.zeros((self.vocab_size, 1))  # 把输入编码成0、1格式，在input中，为0代表此字符未激活，此步骤相当于初始化一个输入向量
            xs[t][inputs[t]] = 1
            # 计算隐藏层
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # RNN的隐藏层神经元激活值计算
            # 计算输出层
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # RNN的输出
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # 概率归一化
            # 计算损失函数(交叉熵损失函数)
            loss += -np.log(ps[t][targets[t], 0])  # softmax 损失函数

        # 反向传播，得到各个权值和偏置的梯度
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(xrange(len(inputs))):  # reversed()反转
            dy = np.copy(ps[t])
            # 计算误差，targets[t]表示t时刻，输出层所输出字的编号，如targets[1]=14,表示“是”
            # 输出层的误差： backprop into y，经过此步骤，dy已经不是原始的dy了，准确应该称为Δdy，即误差：delta_dy = dy
            dy[targets[t]] -= 1

            # 用dy直接做输入，说明从隐藏层到输出层没有用激活函数（不知道我理解的对不对）
            dWhy += np.dot(dy, hs[t].T)  # .T 表示转置  .dot表示矩阵相乘
            dby += dy

            # 隐藏层误差：输出层误差+上一层传回来的误差
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h，

            # 从隐藏层到隐藏层，以及输入层到隐藏层都用了tanh作为激活函数
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity ,tanh函数的导数 [tanh(x)]' = 1-[tanh(x)]^2

            # 计算Whh梯度
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dbh += dhraw

            # 计算Wxh梯度
            dWxh += np.dot(dhraw, xs[t].T)

            # 保存当前时刻隐藏层误差
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # .clip限制元素值在某个范围（-5,5）以内
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    # 预测函数，用于验证，给定seed_ix为t=0时刻的字符索引，生成预测后面的n个字符   是正向传播？？？
    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in xrange(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)  # h是递归更新的
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())  # 根据概率大小挑选
            x = np.zeros((self.vocab_size, 1))  # 更新输入向量
            x[ix] = 1
            ixes.append(ix)  # 保存序列索引
        return ixes

    def run(self):
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0

        while n < 20000:

            # n表示迭代网络迭代训练次数。当输入是t=0时刻时，它前一时刻的隐藏层神经元的激活值我们设置为0
            if p + self.seq_length + 1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))  #
                p = 0  # go from start of data

            # 输入与输出(真实输出)
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p + self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + self.seq_length + 1]]

            # if n == 0:
            # print "[输入]", len(inputs)
            #     for i in inputs:
            #         print '   ', self.ix_to_char[i]
            #
            # print "[输出]", len(targets)
            # for i in targets:
            #         print '   ', self.ix_to_char[i]

            # 当迭代了1000次，
            # 测试输出，跟模型训练基本没关系
            if n % 1000 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)  # inputs[0] = 我，每次都从头开始输出新的预测值，看是否符合
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print 'iter %d, loss: %f' % (n, smooth_loss), len(sample_ix)
                # print '----\n %s \n----' % (txt,)
                # print self.ix_to_char[inputs[0]]
                # print txt

            # 真正训练部分
            # RNN前向传导与反向传播，获取梯度值
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss)  # print progress

            # 采用Adagrad自适应梯度下降法更新三个权值矩阵+两个偏置,
            # 可参看博文：http://blog.csdn.net/danieljianfeng/article/details/42931721
            # 神奇的是没有使用矩阵下标，两行代码就直接修改矩阵的值，，经过探索，原来是numpy.random.rand的原因，具体可自行测试
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam

                # 自适应梯度下降公式
                # 公式可参考 http://blog.csdn.net/heyongluoyao8/article/details/52478715
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            p += self.seq_length  # 批量训练  这句代码较重要，缺少的话，效果较差，具体原因待考证！！！因为P相当于一个滑动窗口
            n += 1  # 记录迭代次数

        # 预测测试：给定一个字，预测一句5个字的句子
        test_hprev = np.zeros((self.hidden_size, 1))
        test_inputs = self.char_to_ix['坤'.decode('utf-8')]
        sample_ix = self.sample(test_hprev, test_inputs, 5)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        print "【测试】", self.ix_to_char[test_inputs]
        print '----\n %s \n----' % (txt,)

        # print "【测试】", self.ix_to_char[inputs[4]]
        # sample_ix = self.sample(test_hprev, inputs[4], 3)
        # txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        # print '----\n %s \n----' % (txt,)
        #
        # print "【测试】", self.ix_to_char[inputs[7]]
        # sample_ix = self.sample(test_hprev, inputs[7], 3)
        # txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        # print '----\n %s \n----' % (txt,)

        # 继续就训练出来的参数写新的预测方法


if __name__ == "__main__":
    r = Rnn()
    r.run()
