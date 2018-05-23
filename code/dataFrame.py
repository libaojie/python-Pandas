#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Comment    : 
@Time       : 2018/5/22 18:41
@Author     : libaojie
@File       : TestDataFarme.py
@Software   : PyCharm
"""

import pandas as pd
from numpy import NaN


def create():
    """
    直接创建
    :return:
    """
    # 直接创建
    df1 = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                       index=[0, 1],
                       columns=['a', 'b', 'c', 'd'])
    # 二维数组转化
    df2 = pd.DataFrame(pd.np.arange(10).reshape(2, 5))
    df3 = pd.DataFrame(pd.np.arange(12).reshape(3, 4), index=['00', '01', '02'],
                       columns=['a', 'b', 'c', 'd'])
    # 随机产生
    df4 = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '02'],
                       columns=['a', 'b', 'c', 'd'])

    print('\ndf1------------------------------------------------------------------------------------------------------')
    print(df1)
    print('\ndf2------------------------------------------------------------------------------------------------------')
    print(df2)
    print('\ndf3------------------------------------------------------------------------------------------------------')
    print(df3)
    print('\ndf4------------------------------------------------------------------------------------------------------')
    print(df4)

    # 字典转化
    # 嵌套列表为一列
    _dict = {'a': ['a0', 'a1'], 'b': ['b0', 'b1']}
    df5 = pd.DataFrame(_dict)
    # 嵌套字典为一列
    _dict = {'a': {0: 'a0', 1: 'a1'}, 'b': {0: 'b0', 1: 'b1'}}
    df6 = pd.DataFrame(_dict)
    # 指定列序
    _dict = {'col1': ['a1', 'b1'], 'col2': ['a2', 'b2']}
    df7 = pd.DataFrame(_dict, columns=['col2', 'col1'])

    print('\ndf5------------------------------------------------------------------------------------------------------')
    print(df5)
    print('\ndf6------------------------------------------------------------------------------------------------------')
    print(df6)
    print('\ndf7------------------------------------------------------------------------------------------------------')
    print(df7)


def attribute():
    """
    属性
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n---------------------------------------------------------------------------------------------------------')
    # 描述
    print(df.describe())
    print('\n---------------------------------------------------------------------------------------------------------')
    # 列类型
    print(df.dtypes)
    print('\n---------------------------------------------------------------------------------------------------------')
    # 行列表
    print(df.index.tolist())
    print('\n---------------------------------------------------------------------------------------------------------')
    # 列列表
    print(df.columns.tolist())
    print('\n---------------------------------------------------------------------------------------------------------')
    # 查看前两行
    print(df.head(2))
    print('\n---------------------------------------------------------------------------------------------------------')
    # 查看后两行
    print(df.tail(2))
    print('\n---------------------------------------------------------------------------------------------------------')
    # 转置
    print(df.T)


def get_value():
    """
    切片筛选值
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=['00', '01'],
                      columns=['a', 'b', 'c', 'd'])
    # 切片筛选值
    print(df)
    print('\n---------------------------------------------------------------------------------------------------------')
    # 取单列得Series
    print(df['b'])
    print('\n---------------------------------------------------------------------------------------------------------')
    # 取单列得DataFrame
    print(df[['b']])
    print('\n---------------------------------------------------------------------------------------------------------')
    # 取多列得DataFrame
    print(df[['b', 'c']])

    print('\n---------------------------------------------------------------------------------------------------------')
    # 行选择
    print(df[0:1])


def loc():
    """
    loc取值
    :return:
    """
    print('\n---------------------------------------------------------------------------------------------------------')
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1'], ['a2', 'b2', 'c2', 'd2']],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n-----------------------------------')
    # 选取行
    print(df.loc[:'00', :])
    print('\n-----------------------------------')
    print(type(df.loc['00']))
    print(df.loc['00'])
    print('\n-----------------------------------')
    print(df.loc[['00', '02']])
    print('\n-----------------------------------')
    # 选取列
    print(df.loc[:, 'b':'d'])

    print('\n-----------------------------------')
    # 行列混合
    print(df.loc['02', 'b'])


def value():
    """
    值
    :return:
    """
    print('\n---------------------------------------------------------------------------------------------------------')
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n-----------------------------------')
    # 筛选某列大于某值的剩余行
    print(df[df['b'] > 6])
    print('\n-----------------------------------')
    # 筛选某列值在某集合中的剩余行
    print(df[df['b'].isin([6, 10])])
    print('\n-----------------------------------')
    # 所有大于6 的值
    print(df[df > 6])
    print('\n-----------------------------------')
    # lambda
    print(df[lambda df: df['b'] == 6])


def insert():
    """
    增加
    :return:
    """

    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n-----------------------------------')
    # 增加一列为相同值
    df['e'] = 'new'
    print(df)


def add():
    """
    相加
    :return:
    """
    df1 = pd.DataFrame([1, 1, 1, pd.np.nan], index=['a', 'b', 'c', 'd'],
                     columns=['one'])
    df2 = pd.DataFrame(dict(one=[1, pd.np.nan, 1, pd.np.nan],
                          two=[pd.np.nan, 2, pd.np.nan, 2]),
                     index=['a', 'b', 'd', 'e'])
    print('\ndf1-----------------------------------')
    print(df1)
    print('\ndf2-----------------------------------')
    print(df2)
    print('\nadd-----------------------------------')
    print(df1.add(df2))
    print('\n--------------------------------------')
    print(df1.add(df2, fill_value=0))
    print('\n+--------------------------------------')


def append():
    df1 = pd.DataFrame([[1, 2, 3, 4], [10, 23, 3, 4], [9, 107, 11, 12]],
                       index=['00', '01', '02'],
                       columns=['a', 'b', 'c', 'd'])

    df2 = pd.DataFrame([[3, 2, 3, 4], [1, 3, 3, 4], [9, 10, 114, 12]],
                       index=['00', '01', '03'],
                       columns=['a', 'b', 'c', 'd'])
    print('\ndf1-----------------------------------')
    print(df1)
    print('\ndf2-----------------------------------')
    print(df2)
    print('\nadd-----------------------------------')
    print(df1.add(df2))
    print('\nadd--------------------------------------')
    print(df1.add(df2, fill_value=0))
    print('\n+--------------------------------------')
    print(df1+df2)
    print('\nappend--------------------------------------')
    print(df1.append(df2))
    print('\nappend--------------------------------------')
    print(df1.append(df2, ignore_index=True))


def concat():
    """
    连接
    :return:
    """
    df1 = pd.DataFrame([[1, 2, 3, 4], [10, 23, 3, 4], [9, 107, 11, 12]],
                       index=['00', '01', '02'],
                       columns=['a', 'b', 'c', 'd'])

    df2 = pd.DataFrame([[3, 2, 3, 4], [1, 3, 3, 4], [9, 10, 114, 12]],
                       index=['00', '01', '03'],
                       columns=['a', 'b', 'c', 'd'])
    print('\ndf1-----------------------------------')
    print(df1)
    print('\ndf2-----------------------------------')
    print(df2)
    print('\nconcat--------------------------------------')
    print(pd.concat([df1, df2]))
    print('\nconcat--------------------------------------')
    print(pd.concat([df1, df2], axis=1))
    print('\nconcat--------------------------------------')
    print(pd.concat([df1, df2], axis=1, join='inner'))


def delete():
    """
    删除
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\ndel--------------------------------------')
    # 真删除、无法多列删
    del df['a']
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
    print('\n--------------------------------------')
    # 移除行
    print(df.drop(['00', '01']))
    print('\n--------------------------------------')
    print(df.drop(index=['00', '02']))
    print('\n--------------------------------------')

    # 移除列
    print(df.drop(['a', 'b'], axis=1))
    print('\n--------------------------------------')
    df.drop(columns=['a', 'c'], inplace=True)
    print(df)


def dropna_row():
    """
    去空 行
    :return:
    """
    df = pd.DataFrame([[1, 2, NaN, 4], [5, 6, 7, 8], [9, 10, NaN, NaN], [NaN, NaN, NaN, NaN]],
                      index=['00', '01', '02', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\nhow=any--------------------------------------')
    # 删除任意有丢失的行
    print(df.dropna(how='any'))
    print('\nhow=all--------------------------------------')
    # 删除全丢失的行
    print(df.dropna(how='all'))
    print('\nthresh--------------------------------------')
    # 保持一行最多有几个NaN
    print(df.dropna(thresh=2))
    print('\nsubset--------------------------------------')
    # 删除某些列任意有NaN
    print(df.dropna(subset=['a', 'd']))
    print('\n--------------------------------------')
    # 删除某些列全部为NaN
    print(df.dropna(subset=['a', 'd'], how='all'))
    pass


def dropna_col():
    """
    去空 列
    :return:
    """
    df = pd.DataFrame([[1, 2, NaN, NaN], [5, 6, 7, NaN], [9, 10, NaN, NaN], [13, NaN, NaN, NaN]],
                      index=['00', '01', '02', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\nhow=any--------------------------------------')
    # 删除任意有丢失的列
    print(df.dropna(how='any', axis=1))
    print('\nhow=all--------------------------------------')
    # 删除全丢失的列
    print(df.dropna(how='all', axis=1))
    print('\nthresh--------------------------------------')
    # 保持一列最多有几个NaN
    print(df.dropna(thresh=2, axis=1))
    print('\nsubset--------------------------------------')
    # 删除某些列任意有NaN
    print(df.dropna(subset=['00', '01'], axis=1))
    print('\ninplace--------------------------------------')
    # # 删除某些列全部为NaN
    df.dropna(subset=['00', '01'], how='all', axis=1, inplace=True)
    print(df)


def dropna():
    """
    不同的空值
    :return:
    """
    df = pd.DataFrame([[1, '', 3, 4], [5, 6, 7, None], [9, 10, pd.np.nan, pd.NaT], [None, 14, None, None]],
                      index=['00', '01', '02', '03'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\nrow--------------------------------------')
    print(df.dropna())
    print('\ncol--------------------------------------')
    print(df.dropna(axis=1))


def isna():
    """
    判空
    :return:
    """
    df = pd.DataFrame({'age': [5, 6, NaN],
                       'born': [pd.NaT, pd.Timestamp('1939-05-27'),
                                pd.Timestamp('1940-04-25')],
                       'name': ['Alfred', 'Batman', ''],
                       'toy': [None, 'Batmobile', 'Joker']})
    print(df)
    print('\nisna-------------------------------------')
    # 判断是否为空
    print(df.isna())
    print('\nisnull-------------------------------------')
    print(df.isnull())
    print('\nnotna-------------------------------------')
    print(df.notna())
    print('\nnotnull-------------------------------------')
    print(df.notnull())


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
    print('\n固定值-------------------------------------')
    # 将所有NaN替换为固定值
    print(df.fillna('new'))
    print('\n不同值-------------------------------------')
    # 将NaN值不同行改为不同值
    values = {'A': 'a_new', 'B': 'b_new', 'C': 'c_new', 'D': 'd_new'}
    print(df.fillna(value=values))
    print('\nlimit-------------------------------------')
    # 修改固定数量
    print(df.fillna(value=values, limit=2))

    print('\nffill 从上往下-------------------------------------')
    # 将NaN值改为上一个值 从上往下
    print(df.fillna(method='ffill'))
    print('\n-------------------------------------')
    # 等价
    print(df.ffill())
    print('\nbfill 从下往上-------------------------------------')
    # 将NaN值改为下一个值
    print(df.fillna(method='bfill'))
    print('\n-------------------------------------')
    # 等价
    print(df.bfill())

    print('\nffill 从左到右-------------------------------------')
    # 从左到右，替换NaN
    print(df.ffill(axis=1))
    print('\nbfill 从右到左-------------------------------------')
    # 从右到左，替换NaN
    print(df.bfill(axis=1))
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
    print('\n整行去重-------------------------------------')
    # 整行去重
    print(df.drop_duplicates())
    print('\nsubset-------------------------------------')
    # 选某几行去重
    print(df.drop_duplicates(subset=['a', 'b']))
    print('\nkeep=first-------------------------------------')
    # 重复默认保持第一行
    print(df.drop_duplicates(keep='first'))
    print('\nkeep=last-------------------------------------')
    # 重复保持最后一行
    print(df.drop_duplicates(keep='last', subset=['a', 'b']))
    print('\nkeep=False-------------------------------------')
    # 删除所有重复项
    df.drop_duplicates(keep=False, inplace=True)
    print(df)


def sort_value():
    """
    按值排序
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '03'],
                      columns=['a', 'b', 'c', 'd'])

    print(df)
    print('\n升序-------------------------------------')
    # 默认升序 先排a列，a列相同排b列
    print(df.sort_values(by=['a', 'b']))
    print('\n降序-------------------------------------')
    # 降序排
    print(df.sort_values(by='a', ascending=False))


def sort_index():
    """
    按轴排序
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['03', '02', '01'],
                      columns=['d', 'b', 'a', 'c'])
    print(df)
    print('\n列索引-------------------------------------')
    print(df.sort_index())
    print('\n行索引-------------------------------------')
    print(df.sort_index(axis=1))


def rename():
    """
    换名
    :return:
    """
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n列索引-------------------------------------')
    # 列换名
    df.rename(columns={'a': 'a1', 'b': 'b1'}, inplace=True)
    print(df)
    print('\nstr-------------------------------------')
    # 应用str
    df.rename(str.upper, axis='columns', inplace=True)
    print(df)
    print('\n行索引-------------------------------------')
    # 行换名
    df.rename({'01': '001', '02': '002'}, axis='index', inplace=True)
    print(df)


def isin1():
    """
    是否包含
    :return:
    """
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    print(df)
    print('-------------------------------------')
    print(df.isin([1, 3, 12, 'a']))
    pass


def isin2():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
    print(df)
    print('-------------------------------------')
    print(df.isin({'A': [1, 3], 'B': [4, 7, 12]}))
    pass


def isin3():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    print(df)
    print('-------------------------------------')
    other = pd.DataFrame({'A': [1, 3, 3, 2], 'B': ['e', 'f', 'f', 'e']})
    print(other)
    print('-------------------------------------')
    print(df.isin(other))
    pass


def where():
    """
    判断
    :return:
    """
    df = pd.DataFrame(pd.np.arange(10).reshape(-1, 2), columns=['A', 'B'])
    m = df % 3 == 0
    print(df)
    print('\nwhere-------------------------------------')
    # 条件成立不修改
    print(df.where(m, -df))
    print('\n-------------------------------------')
    # 等价
    print(df.where(m, -df) == pd.np.where(m, df, -df))
    print('\nmask-------------------------------------')
    # 取反
    print(df.where(m, -df) == df.mask(~m, -df))


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
    print('\n-------------------------------------')
    # 旧值
    print(df.replace(3, 33))


def update():
    """
    修改
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1']],
                      index=[0, 1],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n一个值-------------------------------------')
    # 具体值
    df1 = df.copy()
    df1['b'][1] = 'new'
    print(df1)

    print('\n一列值-------------------------------------')
    # 修改一列为同一值
    df2 = df.copy()
    df2['c'] = 'new'
    print(df2)

    print('\n一行值-------------------------------------')
    # 修改一行为同一值
    df3 = df.copy()
    df3[:1] = 'new'
    print(df3)

    print('\n一列值-------------------------------------')
    # 修改一列为Series
    df4 = df.copy()
    df4['b'] = pd.Series(['new1', 'new2', 'new3'])
    print(df4)

    print('\n-------------------------------------')
    df5 = df.copy()
    df5 = df5['b'].str.upper()
    print(type(df5))
    print(df5)

    print('\n-------------------------------------')
    df['b'] = df['b'].str.upper()
    print(df)
    pass


def set_index():
    """
    将某列作为index
    :return:
    """
    df = pd.DataFrame({'month': [1, 4, 7, 10],
                       'year': [2012, 2014, 2013, 2014],
                       'sale': [55, 40, 84, 31]})
    print(df)
    print('\n-------------------------------------')
    # 将一列作为index
    print(df.set_index('month'))
    print('\n-------------------------------------')
    # 将几列组合作为index
    print(df.set_index(['month', 'year']))
    print('\n-------------------------------------')
    # 将index重新赋值
    print(df.set_index([[1, 2, 3, 4], 'year']))


def reset_index1():
    """
    将index作为一列
    :return:
    """
    df = pd.DataFrame([('bird', 389.0),
                       ('bird', 24.0),
                       ('mammal', 80.5),
                       ('mammal', NaN)],
                      index=['falcon', 'parrot', 'lion', 'monkey'],
                      columns=('class', 'max_speed'))

    print(df)
    print('\n-------------------------------------')
    # 将index作为列名为index的一列
    print(df.reset_index())
    print('\n-------------------------------------')
    # 将index去掉并改为索引
    print(df.reset_index(drop=True))


def reset_index2():
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
    print('\n-------------------------------------')
    print(df.reset_index())
    print('\n-------------------------------------')
    print(df.reset_index(level='class'))
    print('\n-------------------------------------')
    print(df.reset_index(level='class', col_level=1))
    print('\n-------------------------------------')
    print(df.reset_index(level='class', col_level=1, col_fill='species'))
    print('\n-------------------------------------')
    print(df.reset_index(level='class', col_level=1, col_fill='genus'))


def groupby():
    """
    统计数量
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n-------------------------------------')
    print(df.groupby(by='a').count())
    print('\n-------------------------------------')
    print(df.groupby(by='a').mean())
    print('\n-------------------------------------')
    print(df.groupby(by=['a', 'b', 'c', 'd']).count())
    print('\n-------------------------------------')
    print(df.groupby(by=['a', 'b', 'c'])['d'].mean())
    print('\n统计数量-------------------------------------')
    col = df.columns.tolist()
    df['count'] = 1
    df = df.groupby(col).count()
    print(df)
    print('\n拆解index-------------------------------------')
    print(df.reset_index())


def shape():
    """
    一些属性
    :return:
    """

    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\nshape-------------------------------------')
    print(type(df.shape))
    print(df.shape)
    print('\naxes-------------------------------------')
    print(type(df.axes))
    print(df.axes)
    print('\nndim-------------------------------------')
    print(type(df.ndim))
    print(df.ndim)
    print('\nsize-------------------------------------')
    print(type(df.size))
    print(df.size)


def set_axis():
    """
    设置索引
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n列索引-------------------------------------')
    print(df.set_axis(['I', 'II', 'III', 'IIII'], axis=1, inplace=False))
    print('\n行索引-------------------------------------')
    print(df.set_axis(['I', 'II', 'III'], axis=0, inplace=False))


def run():
    print('\n---------------------------------------------------------------------------------------------------------')

    # 设置索引
    # set_axis()

    # shape()

    # 创建
    # create()

    # 属性
    # attribute()

    # 切片
    # get_value()

    # loc取值
    # loc()

    # 值
    # value()

    # 增加
    # insert()

    # 相加
    # add()
    # append()
    # concat()

    # 删除
    # delete()

    # 移除
    # drop()

    # 去空
    # dropna_row()
    # dropna_col()
    # dropna()

    # 判空
    # isna()

    # 填空
    # fillna()

    # 去重
    # drop_duplicates()

    # 按值排序
    # sort_value()

    # 按轴排序
    # sort_index()

    # 换名
    # rename()

    # 包含
    # isin1()
    # isin2()
    # isin3()

    # 判断
    # where()

    # 替换
    # replace()

    # 更新
    # update()

    # 列转index
    # set_index()

    # index转列
    # reset_index1()
    # reset_index2()

    # 统计数量
    # groupby()

    print('\n---------------------------------------------------------------------------------------------------------')
    pass


run()
