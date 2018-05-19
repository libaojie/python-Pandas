#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Comment    : 
@Time       : 2018/5/19 21:09
@Author     : libaojie
@File       : dataFrame.py
@Software   : PyCharm
"""

import pandas as pd


def create_DataFrame():
    """
    创建DataFrame
    :return:
    """
    # 直接创建
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])

    # 字典转化
    # 嵌套列表为一列
    _dict = {'a': ['a0', 'a1'], 'b': ['b0', 'b1']}
    df = pd.DataFrame(_dict)
    # 嵌套字典为一列
    _dict = {'a': {0: 'a0', 1: 'a1'}, 'b': {0: 'b0', 1: 'b1'}}
    df = pd.DataFrame(_dict)
    # 指定列序
    # _dict = {'col1': ['a1', 'b1'], 'col2': ['a2', 'b2']}
    # df = pd.DataFrame(_dict, columns=['col2', 'col1'])

    print(df)


def add_col():
    """
    增加一列
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])
    df['e'] = 'new'
    print(df)


def del_col():
    """
    删除一行
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])
    del df['a']
    print(df)


def update_val():
    """
    修改值
    :return:
    """

    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])

    df['b'] = df['b'].str.upper()
    # 修改一列为Series
    # df['b'] = pd.Series(['new1', 'new2', 'new3'])

    # 修改一行为同一值
    # df[:1] = 'new'
    # 修改一列为同一值
    # df['c'] = 'new'
    # 具体值
    # df['b'][1] = 'new'
    print(df)


def get_value():
    """
    筛选值
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=['00', '01'],
                      columns=['a', 'b', 'c', 'd'])
    # 行选择
    print(df[0:1])

    # # loc选取行
    # print(df.loc[:0,:])
    # # loc选取列
    # print(df.loc[:,'b':'d'])

    # # 取单列得Series
    # print(df['b'])
    # # 取单列得DataFrame
    # print(df[['b']])
    # # 取多列得DataFrame
    # print(df[['b', 'c']])

def get_bool():
    """
    已布尔值筛选
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    # 筛选某列大于某值的剩余行
    print(df[df['b'] > 6])
    # 筛选某列值在某集合中的剩余行
    print(df[df['b'].isin([6, 10])])
    # 所有大于6 的值
    print(df[df > 6])



# 创建
# create_DataFrame()

# 增加一列
# add_col()


# 删除一列
# del_col()

# 修改值
# update_val()

# 获取值
# get_value()

# 筛选
get_bool()