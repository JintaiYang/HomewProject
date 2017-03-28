#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:38:08 2017

@author: yjt
"""

def get_dataset():
    data = []
    for root, dirs, files in os.walk(r'E:\研究生阶段课程作业\python\好玩的数据分析\朴素贝叶斯文本分类\tokens\neg'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'bad'))
    for root, dirs, files in os.walk(r'E:\研究生阶段课程作业\python\好玩的数据分析\朴素贝叶斯文本分类\tokens\pos'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'good'))
    random.shuffle(data)

