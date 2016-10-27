# -*-coding:utf-8 -*-

######################################################
#
#
######################################################
import tensorflow as tf
import numpy as np


# 用TF进行简单的二元回归拟合
def run_regression():
    # 构造数据集
    x1_data = np.random.rand(100).astype(np.float32)
    x2_data = np.random.rand(100).astype(np.float32)
    y_data = x1_data * 10 + x2_data * 5 + 3 + tf.random_uniform([100], -0.1, 0.1)

    # 定义变量：占位符
    w1 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
    w2 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
    b = tf.Variable(tf.zeros([1]))
    y = w1 * x1_data + w2 * x2_data + b

    # loss函数
    loss = tf.reduce_mean(tf.square(y - y_data))
    # 选择优化方法
    optimizer = tf.train.AdadeltaOptimizer(0.6)
    # 利用优化方法对loss函数进行参数估计
    train = optimizer.minimize(loss)

    # 变量初始化
    init = tf.initialize_all_variables()

    # 构造执行器，并执行初始化
    sess = tf.Session()
    sess.run(init)

    for step in range(30001):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(w1), sess.run(w2), sess.run(b), sess.run(loss)

    print step, sess.run(w1), sess.run(w2), sess.run(b), sess.run(loss)


# 计时器
def run_counter():
    # 设计图
    state = tf.Variable(0, name='counter')

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)  # 此操作相当于将新值赋给state，类似于操作 state = new_value

    init_op = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init_op)

    # 打印结果
    print sess.run(state)

    for _ in range(3):
        sess.run(update)
        print sess.run(state)


# 测试fetch的用法
def run_fetch():
    input_1 = tf.constant(3.0)
    input_2 = tf.constant(2.0)
    input_3 = tf.constant(5.0)

    add = tf.add(input_1, input_2)  # 5
    mul = tf.mul(input_3, add)  # 25

    sess = tf.Session()

    print sess.run(add)
    print sess.run([add, mul])
    print sess.run([mul])


if __name__ == "__main__":
    run_fetch()
