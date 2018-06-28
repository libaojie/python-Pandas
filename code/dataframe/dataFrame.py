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
from numpy.random.mtrand import randn


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


def attribute1():
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


def attribute2():
    df = pd.DataFrame(pd.np.random.randn(50, 4), columns=list('ABCD'))
    print(df)
    print('\n前5行-----------------------------------')
    print(df.head(5))
    print('\n后5行-----------------------------------')
    print(df.tail(5))
    print('\n随机5行-----------------------------------')
    print(df.sample(n=5))
    print('\n随机抽10%-----------------------------------')
    print(df.sample(frac=0.1, replace=True))


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
    print(df1 + df2)


def append():
    df1 = pd.DataFrame([[1, NaN, 3], [NaN, 10, NaN], [9, 107, 11]],
                       index=['00', '01', '02'],
                       columns=['a', 'b', 'c'])

    df2 = pd.DataFrame([[3, 2, 3], [1, NaN, 3], [9, 10, NaN]],
                       index=['00', '01', '03'],
                       columns=['a', 'b', 'd'])
    print('\ndf1-----------------------------------')
    print(df1)
    print('\ndf2-----------------------------------')
    print(df2)
    print('\nadd-----------------------------------')
    print(df1.add(df2))
    print('\nadd--------------------------------------')
    print(df1.add(df2, fill_value=1))
    print('\n+--------------------------------------')
    print(df1 + df2)
    print('\nappend--------------------------------------')
    print(df1.append(df2))
    print('\nappend--------------------------------------')
    print(df1.append(df2, ignore_index=True))
    print('\nconcat--------------------------------------')
    print(pd.concat([df1, df2]))
    print('\nconcat--------------------------------------')
    print(pd.concat([df1, df2], axis=1))
    print('\nupdate--------------------------------------')
    print(df1.update(df2))


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

    print('\n-------------------------------------')
    df = pd.DataFrame({
        'col1': ['A', 'A', 'B', pd.np.nan, 'D', 'C'],
        'col2': [2, 1, 9, 8, 7, 4],
        'col3': [0, 1, 9, 4, 2, 3],
    })
    print(df)
    print('\n-------------------------------------')
    print(df.sort_values(by='col1', ascending=False))
    print('\n-------------------------------------')
    print(df.sort_values(by='col1', ascending=False, na_position='first'))


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


def rename_axis():
    """

    :return:
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    print(df)
    print('\n行索引取名-------------------------------------')
    print(df.rename_axis("foo"))
    print('\n列索引取名-------------------------------------')
    print(df.rename_axis("bar", axis="columns"))


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
    # print('\n-------------------------------------')
    # print(df.groupby('a').count())
    # print('\n-------------------------------------')
    # print(df.groupby(by='a').mean())
    # print('\n-------------------------------------')
    # print(df.groupby(by=['a', 'b'])['d'].count())
    # print('\n-------------------------------------')
    # print(df.groupby(by=['a', 'b'])['c', 'd'].mean())
    # # print('\n统计数量-------------------------------------')
    col = df.columns.tolist()
    df['count'] = 1
    df = df.groupby(col).count()
    # 拆解index
    print(df.reset_index())
    # print(df)
    # # print('\n拆解index-------------------------------------')
    #


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


def pop():
    """
    剔除
    :return:
    """
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [9, 10, 11, 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print(df)
    print('\n-------------------------------------')
    print(df.pop('b'))
    print('\n-------------------------------------')
    print(df)


def squeeze():
    df = pd.DataFrame(pd.np.random.randn(3, 4), index=['03', '02', '01'],
                      columns=['d', 'b', 'a', 'c'])
    print(df)
    print('\n-------------------------------------')
    print(df.squeeze(axis=0))


def equals():
    """
    是否相等
    :return:
    """
    df1 = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]],
                       index=['00', '01'],
                       columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]],
                       index=['00', '01'],
                       columns=['a', 'b', 'c', 'd'])
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n每个值比较-------------------------------------')
    print(df1 == df2)
    print('\n所有值比较-------------------------------------')
    print(df1.equals(df2))


def keys():
    df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]],
                      index=['00', '01'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\nkeys-------------------------------------')
    print(type(df.keys))
    print(df.keys)
    # print('\niteritems-------------------------------------')
    # print(type(df.iteritems))
    # for i in df.iteritems:
    #     print(i)


def empty():
    """
    是否为空
    :return:
    """
    df1 = pd.DataFrame({'A': []})
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf1.empty-------------------------------------')
    print(df1.empty)

    print('\ndf2-------------------------------------')
    df2 = pd.DataFrame({'A': [pd.np.nan]})
    print(df2)
    print('\ndf2.empty-------------------------------------')
    print(df2.empty)
    print('\n-------------------------------------')
    print(df2.dropna().empty)
    pass


def to_json():
    """
    转化为json
    :return:
    """
    df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
    print('\ndf-------------------------------------')
    print(df)
    print('\nindex-------------------------------------')
    print(df.to_json(orient='index'))
    print('\nrecords-------------------------------------')
    print(df.to_json(orient='records'))
    print('\ntable-------------------------------------')
    print(df.to_json(orient='table'))


def take():
    df = pd.DataFrame([('falcon', 'bird', 389.0),
                       ('parrot', 'bird', 24.0),
                       ('lion', 'mammal', 80.5),
                       ('monkey', 'mammal', pd.np.nan)],
                      columns=('name', 'class', 'max_speed'),
                      index=[0, 2, 3, 1])
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.take([0, 3]))
    print('\n-------------------------------------')
    print(df.take([1, 2], axis=1))
    print('\n-------------------------------------')
    print(df.take([-1, -2]))


def xs():
    # 只能get不能set
    df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(type(df.xs('row 1')))
    print(df.xs('row 1'))
    print('\n-------------------------------------')
    print(type(df.xs('col 1', axis=1)))
    print(df.xs('col 1', axis=1))


def add_prefix():
    """
    列索引加值
    :return:
    """
    df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row1', 'row2'], columns=['col1', 'col2'])
    print('\ndf-------------------------------------')
    print(df)
    print('\n前缀-------------------------------------')
    print(df.add_prefix('new'))
    print('\n后缀-------------------------------------')
    print(df.add_suffix('new'))


def reindex1():
    """
    保留或新添索引
    :return:
    """
    index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
    df = pd.DataFrame({
        'http_status': [200, 200, 404, 404, 301],
        'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
        index=index)
    print('\ndf-------------------------------------')
    print(df)
    print('\n处理行索引-------------------------------------')
    new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
                 'Chrome']
    print(df.reindex(new_index))
    print('\n新行填充0-------------------------------------')
    print(df.reindex(new_index, fill_value=0))
    print('\n新行填充Nan-------------------------------------')
    print(df.reindex(new_index, fill_value='missing'))

    print('\n处理列-------------------------------------')
    print(df.reindex(columns=['http_status', 'user_agent']))
    print('\n-------------------------------------')
    print(df.reindex(['http_status', 'user_agent'], axis="columns"))


def reindex2():
    date_index = pd.date_range('1/1/2010', periods=6, freq='D')
    df2 = pd.DataFrame({"prices": [100, 101, pd.np.nan, 100, 89, 88]},
                       index=date_index)
    print('\ndf-------------------------------------')
    print(df2)

    print('\n-------------------------------------')
    date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
    print(df2.reindex(date_index2))
    print('\n-------------------------------------')
    print(df2.reindex(date_index2, method='bfill'))


def filter():
    """
    过滤
    :return:
    """
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      index=['mouse', 'rabbit', 'rt'],
                      columns=['one', 'two', 'three'])
    print('\ndf-------------------------------------')
    print(df)
    print('\nitems-------------------------------------')
    print(df.filter(items=['one', 'three']))
    print('\nregex-------------------------------------')
    print(df.filter(regex='e$', axis=1))
    print('\nlike-------------------------------------')
    print(df.filter(like='bbi', axis=0))


def transform():
    df = pd.DataFrame(pd.np.random.randn(6, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=6))
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    df.iloc[2:4] = pd.np.nan
    print('\n-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.transform(lambda x: (x - x.mean()) / x.std()))


def math1():
    """
    数学统计
    :return:
    """
    df = pd.DataFrame([[1, 2, NaN, 4], [5, NaN, NaN, 12]],
                      index=['00', '01'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\n求列和-------------------------------------')
    print(df.sum())
    print('\n求行和-------------------------------------')
    print(df.sum(axis=1))

    print('\n求累加-------------------------------------')
    print(df.cumsum())
    print(df.cumsum(axis=1))

    print('\n求最大值所在行-------------------------------------')
    print(df.idxmax())
    print('\n求最大值所在列-------------------------------------')
    print(df.idxmax(axis=1))


def math2():
    df = pd.DataFrame([[1, 2, NaN, 4], [5, NaN, NaN, 12], [1, 2, NaN, 4]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\n非Nan数量-------------------------------------')
    print(df.count())
    print('\n-------------------------------------')
    print(df.count(axis=1))
    print('\nnunique-------------------------------------')
    print(df.nunique())
    print('\nnunique-------------------------------------')
    print(df.nunique(axis=1))
    print('\n单列去重-------------------------------------')
    print(type(df['a'].unique()))
    print(df['a'].unique())
    print('\n单列统计数量-------------------------------------')
    print(type(df['a'].value_counts()))
    print(df['a'].value_counts())


def items():
    df = pd.DataFrame([[1, 2, NaN, 4], [5, NaN, NaN, 12], [1, 2, NaN, 4]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\nitems-------------------------------------')
    for col_key, col_val in df.items():
        for row_key, row_val in col_val.items():
            print(' [{0},{1}]:{2}'.format(col_key, row_key, row_val))

    print('\niteritems-------------------------------------')
    for col_key, col_val in df.iteritems():
        for row_key, row_val in col_val.iteritems():
            print('[{0},{1}]:{2}'.format(col_key, row_key, row_val))


def to_dict():
    df = pd.DataFrame([[1, 2, NaN, 4], [5, NaN, NaN, 12]],
                      index=['00', '01'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.to_dict())
    print('\nseries-------------------------------------')
    print(df.to_dict('series'))
    print('\nsplit-------------------------------------')
    print(df.to_dict('split'))
    print('\nrecords-------------------------------------')
    print(df.to_dict('records'))
    print('\nindex-------------------------------------')
    print(df.to_dict('index'))
    pass


def query():
    df = pd.DataFrame(randn(10, 2), columns=list('ab'))
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.query('a > b'))
    print('\n-------------------------------------')
    print(df[df.a > df.b])


def eval():
    df = pd.DataFrame(randn(10, 2), columns=list('ab'))
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.eval('a + b'))
    print('\n-------------------------------------')
    print(df.eval('c = a + b'))


def select_dtypes():
    df = pd.DataFrame({'a': pd.np.random.randn(6).astype('f4'),
                       'b': [True, False] * 3,
                       'c': [1.0, 2.0] * 3,
                       'd': [100] * 6})
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.select_dtypes(include='bool'))
    print('\n-------------------------------------')
    print(df.select_dtypes(include=['float64']))
    print('\n-------------------------------------')
    print(df.select_dtypes(exclude=['floating']))
    pass


def assign():
    df = pd.DataFrame({'A': range(1, 6), 'B': pd.np.random.randn(5)})
    print('\ndf-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.assign(ln_A=lambda x: pd.np.log(x.A)))


def nlargest():
    df = pd.DataFrame({'a': [1, 10, 8, 11, -1],
                       'b': list('abdce'),
                       'c': [1.0, 2.0, pd.np.nan, 3.0, 4.0]})
    print('\ndf-------------------------------------')
    print(df)
    print('\n以某列值降序取前3个-------------------------------------')
    print(df.nlargest(3, 'a'))
    print(df.nsmallest(3, 'a'))


def combine():
    df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
    df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
    print('\n-------------------------------------')
    print(df1)
    print('\n-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    # print(df1.combine(df2, lambda s1, s2: s1 if s1.sum() < s2.sum() else s2))
    print(df1.combine(df2, lambda s1, s2: s1 if s1.sum() < s2.sum() else s2))


def combine_first():
    df1 = pd.DataFrame([[1, pd.np.nan], [NaN, 3]])
    df2 = pd.DataFrame([[3, 4, 10], [5, NaN, NaN], [6, 7, 9]])
    print('\n-------------------------------------')
    print(df1)
    print('\n-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    print(df1.combine_first(df2))


def update1():
    df1 = pd.DataFrame({'A': [1, 2, 3],
                        'B': [400, 500, 600]})
    df2 = pd.DataFrame({'B': [4, 5, 6],
                        'C': [7, 8, 9]})
    print('\n-------------------------------------')
    print(df1)
    print('\n-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    df1.update(df2)
    print(df1)


def update2():
    df1 = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    df2 = pd.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']})
    df3 = pd.DataFrame({'B': [4, pd.np.nan, 6]})
    df4 = pd.DataFrame({'B': ['d', 'e']}, index=[1, 2])
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    df1.update(df2)
    print(df1)
    print('\ndf3-------------------------------------')
    print(df3)
    print('\n-------------------------------------')
    df1.update(df3)
    print(df1)
    print('\ndf4-------------------------------------')
    print(df4)
    print('\n-------------------------------------')
    df1.update(df4)
    print(df1)


def pivot():
    df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'baz': [1, 2, 3, 4, 5, 6]})
    print('\n-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.pivot(index='foo', columns='bar', values='baz'))
    print('\n-------------------------------------')
    print(df.pivot(index='foo', columns='bar')['baz'])


def stack():
    df = pd.DataFrame([[1, 2], [5, NaN]],
                      index=['00', '01'],
                      columns=['a', 'b'])
    print('\n-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    print(df.stack())


def applymap():
    df = pd.DataFrame(pd.np.random.randn(3, 3))
    print('\n-------------------------------------')
    print(df)
    print('\n-------------------------------------')
    df = df.applymap(lambda x: '%.2f' % x)
    print(df)


def join():
    df1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
    df2 = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                        'B': ['B0', 'B1', 'B2']})
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    print(df1.join(df2, lsuffix='_caller', rsuffix='_other'))
    print('\n-------------------------------------')
    print(df1.set_index('key').join(df2.set_index('key')))
    print('\n-------------------------------------')
    print(df1.join(df2.set_index('key'), on='key'))


def nunique1():
    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    # print('\ndf-------------------------------------')
    # print(df)
    # print('\n逐列统计不重复项数量-------------------------------------')
    # print(df.nunique())
    # print('\n逐行统计不重复项数量-------------------------------------')
    # print(df.nunique(axis=1))

    df = pd.DataFrame({'A': [1, NaN, 3], 'B': [1, 1, 1]})
    print('\ndf-------------------------------------')
    print(df)
    print('\n逐列统计非空数量-------------------------------------')
    print(df.count())
    print('\n逐行统计非空数量-------------------------------------')
    print(df.count(axis=1))


def nunique():
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b0', 'c1', 'd1'], ['a2', 'b2', 'c0', 'd2']],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-------------------------------------')
    print(df)
    print('\nnunique-------------------------------------')
    print(df.nunique())
    print('\ncount-------------------------------------')
    print(df.count())
    print('\nmode-------------------------------------')
    print(df.mode())


def merge2():
    df1 = pd.DataFrame([[1, 2, 3], [10, 11, 12], [100, 111, 121]], index=['row1', 'row2', 'row3'],
                       columns=['col1', 'col2', 'col3'])
    df2 = pd.DataFrame([[1, 2, 3], [10, 11, 13], [100, 111, 121]], index=['row1', 'row2', 'row4'],
                       columns=['col1', 'col2', 'col3'])
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n交集-------------------------------------')
    df3 = pd.merge(df1, df2, how='inner')
    print(df3)
    print('\n追加-------------------------------------')
    df4 = df1.append(df3)
    print(df4)
    print('\n去重-------------------------------------')
    df5 = df4.drop_duplicates(keep=False)
    print(df5)


def merge1():
    df1 = pd.DataFrame([[1, 2, 3], [10, 11, 12], [100, 110, 121]], index=['row1', 'row2', 'row3'],
                       columns=['col1', 'col2', 'col3'])
    df2 = pd.DataFrame([[1, 2, 4], [10, 11, 13], [100, 111, 121]], index=['row1', 'row2', 'row4'],
                       columns=['col1', 'col2', 'col4'])
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n有相同的列 整行匹配-------------------------------------')
    print(pd.merge(df1, df2))
    print('\ninner-------------------------------------')
    print(pd.merge(df1, df2, how='inner'))
    print('\nleft-------------------------------------')
    print(pd.merge(df1, df2, how='left'))
    print('\nright-------------------------------------')
    print(pd.merge(df1, df2, how='right'))
    print('\nouter-------------------------------------')
    print(pd.merge(df1, df2, how='outer'))
    print('\non-------------------------------------')
    print(pd.merge(df1, df2, on=['col1']))


def merge_ordered():
    df1 = pd.DataFrame({'key': ['a', 'c', 'e', 'a', 'c', 'e'],
                        'lvalue': [1, 2, 3, 1, 2, 3],
                        'group': ['a', 'a', 'a', 'b', 'b', 'b']})
    df2 = pd.DataFrame({'key': ['b', 'c', 'd'],
                        'rvalue': [1, 2, 3]})
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\ndf-------------------------------------')
    print(pd.merge_ordered(df1, df2, fill_method='ffill', left_by='group'))


def merge_asof():
    df1 = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
    print('\ndf1-------------------------------------')
    print(df1)
    print('\ndf2-------------------------------------')
    print(df2)
    print('\n-------------------------------------')
    print(pd.merge_asof(df1, df2, on='a'))
    print('\n-------------------------------------')
    print(pd.merge_asof(df1, df2, on='a', allow_exact_matches=False))
    print('\n-------------------------------------')
    print(pd.merge_asof(df1, df2, on='a', direction='forward'))
    print('\n-------------------------------------')
    print(pd.merge_asof(df1, df2, on='a', direction='nearest'))

    left = pd.DataFrame({'left_val': ['a', 'b', 'c']}, index=[1, 5, 10])
    right = pd.DataFrame({'right_val': [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    print('\nleft-------------------------------------')
    print(left)
    print('\nright-------------------------------------')
    print(right)
    print('\n-------------------------------------')
    print(pd.merge_asof(left, right, left_index=True, right_index=True))


def run():
    print('\n---------------------------------------------------------------------------------------------------------')
    # nunique1()
    # merge_asof()
    # merge_ordered()
    # merge1()
    # merge2()
    # nunique()
    # join()
    # applymap()
    # stack()
    # pivot()
    # update1()
    # update2()

    # combine_first()
    # combine()
    # nlargest()
    # assign()
    # select_dtypes()
    # eval()

    # query()
    # to_dict()

    # items()

    # math1()
    math2()

    # transform()
    # filter()

    # reindex1()
    # reindex2()

    # add_prefix()

    # xs()

    # take
    # take()

    # 转化json
    # to_json()

    # 判断是否为空
    # empty()

    # keys()

    # 是否相等
    # equals()

    # squeeze
    # squeeze()

    # pop
    # pop()

    # 设置索引
    # set_axis()

    # shape()

    # 创建
    # create()

    # 属性
    # attribute1()
    # attribute2()

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
    # rename_axis()

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
