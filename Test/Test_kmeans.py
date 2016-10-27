# -*- coding:utf-8 -*-

import tensorflow as tf
from random import choice,shuffle
from numpy import array

def kmeans(vectors, noofclusters):
    """
    kmeans聚类
    :param vectors: 应该是一个n *k 的二维numpy的数组，其中n 代表k维向量的数目
    :param noofclusters: 代表了待分的集群数目，是一个整型值
    :return:
    """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    # 找出每个向量的维度
    dim = len(vectors)

    # 辅助随机地从可得到的向量中选取中心点
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    # 计算图
    # 创建一个默认的计算流的图用于整个算法，这样保证了当函数被多次调用时，默认图不会被从上一次调用时留下来的伪造使用的或者Variable挤满
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        ##构建基本的计算单元
        ##首先需要保证每一个中心点都会存在一个Variable矩阵
        ##从现有的点集合中抽取一部分作为默认的中心点
        centroids = [tf.Variable((vectors[vector_indices[i]])) for i in range(noofclusters)]

        ## 创建一个placeholder 用于存放各个中心点可能的分类情况
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assgn(centroid, centroid_value))

        ## 对于每个独立向量的分属的类别设置维默认值 0
        assignments = [tf.Variable(0) for i in range(len(vectors))]

        ## 这些节点在后续的操作中会被分配到合适的值
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment, assignment_value))

        ## 下面创建用于计算平均值的操作节点

        # 输入的placeholder
        mean_input = tf.placeholder("float", [None, dim])

        #节点/op接受输入，并且计算0维度的平均值，譬如输入的向量列表
        mean_op = tf.reduce_mean(mean_input, 0)

        ## 用于计算欧几里得距离的节点
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])

        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2)))

        ##这个op会决定应该将哪个向量归属到哪个节点
        ##基于向量到中心的欧几里得距离

        # placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        ## 初始化所有的状态值
        init_op = tf.initialize_all_variables()

        # 执行初始化操作，初始化所有变量
        sess.run(init_op)

        ## 集群遍历
        # 接下来在k-means聚类迭代中使用最大期望算法，为简单期间，只让它执行固定
        noofiterations = 100
        for iteration_n in range(noofiterations):
            ## 期望步骤
            ## 基于上次迭代后算出的中心点的未知
            ## the_expected_centroid assignments
            ## 首先遍历所有向量
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # 计算给定向量于分配的中心节点之间的欧几里得距离
                distances = [sess.run(euclid_dist, feed_dict={v1:vect, v2:sess.run(centroid)}) for centroid in centroids]

                # 下面可以使用集群分配操作，将上述的距离当做输入
                assignment = sess.run(cluster_assignment, feed_dict={centroid:distances})

                #接下来为每个向量分配合适的值
                sess.run(cluster_assigns[vector_n], feed_dict={centroid_distances:distances})

                ##最大化的步骤
                # 基于上述的期望步骤，计算每个新的中心点的距离从而使集群内的平方和最小
                for cluster_n in range(noofclusters):
                    #收集所有分配给该集群的向量
                    assignment_value
