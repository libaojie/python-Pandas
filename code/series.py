#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Comment    : 
@Time       : 2018/5/19 23:20
@Author     : libaojie
@File       : series.py
@Software   : PyCharm
"""

def create():
    a = pf.Series([1, 2, 3, 4])  # 列表创建

    b = pd.Series(25, index=['a', 'b', 'c'])  # 标量创建

    c = pd.Series({'a': 12, 'b': 23, 'c': 43})  # 字典创建,键为索引

    d = pd.Series(np.arange(5))