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
from numpy import NaN


def create_DataFrame():
    """
    创建DataFrame
    :return:
    """
    # 直接创建
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])

    df = pd.DataFrame(pd.np.arange(10).reshape(2, 5))
    print(df)

    # 随机
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '03'],
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


def attribute():
    """
    属性
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    # 列类型
    print(df.dtypes)
    # 行列表
    print(df.index.tolist())
    # 列列表
    print(df.columns.tolist())
    # 查看前两行
    print(df.head(2))
    # 查看后两行
    print(df.tail(2))
    # 转置
    print(df.T)


def sort_value():
    """
    按值排序
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])

    print(df)
    # 默认升序 先排a列，a列相同排b列
    print(df.sort_values(by=['a', 'b']))
    # 降序排
    print(df.sort_values(by='a', ascending=False))


def sort_index():
    """
    按轴排序
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '03'],
                      columns=['d', 'b', 'a', 'c'])
    print(df)
    print(df.sort_index(axis=1))


def rename():
    """
    列换名
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    # 列换名
    df.rename(columns={'a': 'a1', 'b': 'b1'}, inplace=True)
    print(df)
    # 应用str
    df.rename(str.upper, axis='columns', inplace=True)
    print(df)
    # 行换名
    df.rename({'01': '001', '02': '002'}, axis='index', inplace=True)
    print(df)


def drop():
    """
    移除
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    # 移除行
    print(df.drop(['00', '01']))
    print(df.drop(index=['00', '02']))

    # 移除列
    print(df.drop(['a', 'b'], axis=1))
    print(df.drop(columns=['a', 'c']))


def dropna():
    """
    填充丢失数据
    :return:
    """
    # df = pd.DataFrame([[1, 2, NaN, 4], [5, 6, 7, 8], [9, 10, NaN, NaN], [NaN, NaN, NaN, NaN]],
    #                   index=['00', '01', '02', '03'],
    #                   columns=['a', 'b', 'c', 'd'])
    # print(df)
    # # 删除任意有丢失的行
    # print(df.dropna(how='any'))
    # # 删除全丢失的行
    # print(df.dropna(how='all'))
    # df.dropna(how='all', inplace=True)
    # # 保持一行最多有几个NaN
    # print(df.dropna(thresh=2))
    # # 删除某些列任意有NaN
    # print(df.dropna(subset=['a', 'd']))
    # # 删除某些列全部为NaN
    # print(df.dropna(subset=['a', 'd'], how='all'))

    # df = pd.DataFrame([[1, 2, NaN, NaN], [5, 6, 7, NaN], [9, 10, NaN, NaN], [13, NaN, NaN, NaN]],
    #                   index=['00', '01', '02', '03'],
    #                   columns=['a', 'b', 'c', 'd'])
    # print(df)
    # # 删除任意有丢失的列
    # print(df.dropna(how='any', axis=1))
    # # 删除全丢失的列
    # print(df.dropna(how='all', axis=1))
    # df.dropna(how='all', axis=1, inplace=True)
    # # 保持一列最多有几个NaN
    # print(df.dropna(thresh=2, axis=1))
    # # 删除某些列任意有NaN
    # print(df.dropna(subset=['00', '01'], axis=1))
    # # # 删除某些列全部为NaN
    # df.dropna(subset=['00', '01'], how='all', axis=1, inplace=True)
    # print(df)

    # None的效果与NaN等同
    df = pd.DataFrame([[1, '', 3, 4], [5, 6, 7, None], [9, 10, None, None], [None, None, None, None]],
                      index=['00', '01', '02', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print(df.dropna())
    print(df.dropna(axis=1))

    pass


def drop_duplicates():
    """
    去重
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5, 6], [1, 7, 3, 8]],
                      index=['00', '01', '02', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    # 整行去重
    print(df.drop_duplicates())
    # 选某几行去重
    print(df.drop_duplicates(subset=['a', 'b']))
    # 重复默认保持第一行
    print(df.drop_duplicates(keep='first'))
    # 重复保持最后一行
    print(df.drop_duplicates(keep='last', subset=['a', 'b']))
    # 删除所有重复项
    df.drop_duplicates(keep=False, inplace=True)
    print(df)


def fillna():
    """
    填空
    :return:
    """
    df = pd.DataFrame([[NaN, 2, NaN, 0],
                       [3, 4, NaN, 1],
                       [NaN, NaN, NaN, 5],
                       [NaN, 3, NaN, 4]], columns=list('ABCD'))
    print(df)

    # # 从左到右，替换NaN
    # print(df.ffill(axis=1))
    # # 从右到左，替换NaN
    # print(df.bfill(axis=1))

    # # 将NaN值改为前一个值
    # print(df.fillna(method='ffill'))
    # # 等价
    # print(df.ffill())
    # # 将NaN值改为后一个值
    # print(df.fillna(method='bfill'))
    # # 等价
    # print(df.bfill())

    # # 将所有NaN替换为固定值
    # print(df.fillna('new'))
    # # 将NaN值不同行改为不同值
    # values = {'A': 'a_new', 'B': 'b_new', 'C': 'c_new', 'D': 'd_new'}
    # print(df.fillna(value=values))
    # # 修改固定数量
    # print(df.fillna(value=values, limit=2))


def replace():
    """
    替换
    :return:
    """
    df = pd.DataFrame([[NaN, 3, NaN, 0],
                       [3, 3, NaN, 3],
                       [NaN, NaN, NaN, 5],
                       [NaN, 3, NaN, 4]], columns=list('ABCD'))
    print(df)
    # 旧值
    print(df.replace(3, 33))


def set_index():
    """
    将某列作为index
    :return:
    """
    df = pd.DataFrame({'month': [1, 4, 7, 10],
                       'year': [2012, 2014, 2013, 2014],
                       'sale': [55, 40, 84, 31]})
    print(df)
    # 将一列作为index
    print(df.set_index('month'))
    # 将几列组合作为index
    print(df.set_index(['month', 'year']))
    # 将index重新赋值
    print(df.set_index([[1, 2, 3, 4], 'year']))


def reset_index():
    """
    将index作为一列
    :return:
    """
    # df = pd.DataFrame([('bird', 389.0),
    #                    ('bird', 24.0),
    #                    ('mammal', 80.5),
    #                    ('mammal', NaN)],
    #                   index=['falcon', 'parrot', 'lion', 'monkey'],
    #                   columns=('class', 'max_speed'))
    #
    # print(df)
    # # 将index作为列名为index的一列
    # print(df.reset_index())
    # # 将index去掉并改为索引
    # print(df.reset_index(drop=True))

    index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                       ('bird', 'parrot'),
                                       ('mammal', 'lion'),
                                       ('mammal', 'monkey')],
                                      names=['class', 'name'])
    columns = pd.MultiIndex.from_tuples([('speed', 'max'),
                                         ('species', 'type')])

    df = pd.DataFrame([(389.0, 'fly'),
                       (24.0, 'fly'),
                       (80.5, 'run'),
                       (NaN, 'jump')],
                      index=index,
                      columns=columns)
    print(df)
    print('-------------------------------------')
    print(df.reset_index(level='class'))
    print('-------------------------------------')
    print(df.reset_index(level='class', col_level=1))
    print('-------------------------------------')
    print(df.reset_index(level='class', col_level=1, col_fill='species'))
    print('-------------------------------------')
    print(df.reset_index(level='class', col_level=1, col_fill='genus'))


def bool_():
    """
    一些判断
    :return:
    """
    # df = pd.DataFrame({'age': [5, 6, NaN],
    #                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
    #                             pd.Timestamp('1940-04-25')],
    #                    'name': ['Alfred', 'Batman', ''],
    #                    'toy': [None, 'Batmobile', 'Joker']})
    # print(df)
    # print('-------------------------------------')
    # # 判断是否为空
    # print(df.isna())
    # print('-------------------------------------')
    # print(df.isnull())
    # print('-------------------------------------')
    # print(df.notna())
    # print('-------------------------------------')
    # print(df.notnull())
    #
    pass


def isin():
    """
    是否包含
    :return:
    """
    # df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    # print(df)
    # print('-------------------------------------')
    # print(df.isin([1, 3, 12, 'a']))

    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
    # print(df)
    # print('-------------------------------------')
    # print(df.isin({'A': [1, 3], 'B': [4, 7, 12]}))

    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    print(df)
    print('-------------------------------------')
    other = pd.DataFrame({'A': [1, 3, 3, 2], 'B': ['e', 'f', 'f', 'e']})
    print(other)
    print('-------------------------------------')
    print(df.isin(other))


def where():
    """
    判断
    :return:
    """
    df = pd.DataFrame(pd.np.arange(10).reshape(-1, 2), columns=['A', 'B'])
    m = df % 3 == 0
    print(df)
    print('-------------------------------------')
    # 条件成立不修改
    print(df.where(m, -df))
    print('-------------------------------------')
    # 等价
    print(df.where(m, -df) == pd.np.where(m, df, -df))
    print('-------------------------------------')
    # 取反
    print(df.where(m, -df) == df.mask(~m, -df))

# -------------------------------------------------------------------------
# 待整理
# ------------------------------------------------------------------------


def duplicates_count():
    """
    统计重复数量
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('-------------------------------------')
    cols = df.columns.tolist()
    df['count'] = 1

    df = df.groupby(cols).count()
    print(df)
    # print('-------------------------------------')
    # print(df.index.tolist())

    # for i in df.index.tolist():
    #     for j in i:
    #         print(type(j))
    print('-------------------------------------')
    print(df.reset_index(level=['a','b','c','d']))
    print(df.reset_index())

def append():
    """
    相加统计
    :return:
    """
    df1 = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])

    df2 = pd.DataFrame([[3, 2, 3, 4], [1, 3, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])

    print(df1)
    print('-------------------------------------')
    print(df2)
    # print('-------------------------------------')
    # df3 = df1+df2
    # print(df3)

    # print('-------------------------------------')
    # df4 = df1.append(df2)
    # print(df4)
    # print('-------------------------------------')
    # df5 = pd.concat([df1, df2])
    # print(df5)

    print('-------------------------------------')
    df6 = pd.concat([df1, df2], axis=1)
    # df6 = df6.drop(df1.columns.tolist(), axis=1)
    print(df6)
    df6.fillna(0, inplace=True)
    count= len(df6.columns)
    print(df6.ix[:,0:int(count/2)]+df6.ix[:, int(count/2):count])


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
# get_bool()

# 属性
# attribute()

# 按值排序
# sort_value()

# 按轴排序
# sort_index()

# 列换名
# rename()

# 移除
# drop()

# 去空
# dropna()

# 去重
# drop_duplicates()

# 填空
# fillna()

# 替换
# replace()

# 将一列作为index
# set_index()

# 将index作为一列
# reset_index()

# 一些判断
# bool_()

# 是否包含
# isin()

# 判断
# where()

# 统计重复
duplicates_count()

# 相加统计
# append()