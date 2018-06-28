import pandas as pd


def main():
    print('\n---------------------------------------------------------------------------------------------------------')
    # 获取行
    # get_row1()
    # get_row_no_index()

    # get_col()
    # print('\n---------------------------------------------------------------------------------------------------------')

    # get_val()
    # print('\n---------------------------------------------------------------------------------------------------------')

    # get_val1()

    # where1()
    for1()

    pass

def for1():
    df = pd.DataFrame(pd.np.arange(10).reshape(-1, 2), columns=['A', 'B'])
    print('\ndf-----------------')
    print(df)

    for i in range(df.shape[0]):
        # 取行数
        for j in range(df.shape[1]):
            # 取列数
            print(df.iat[i, j])

    def _f(x):
        print(x)
    # 列循环
    df.apply(_f)
    # 行循环
    df.apply(_f, axis=1)
    # 逐元素， 先循环列 后循环行
    df.apply(lambda val: val.apply(_f))
    # 逐元素
    df.applymap(_f)

    pass


def where1():
    """
    判断
    :return:
    """
    df = pd.DataFrame(pd.np.arange(10).reshape(-1, 2), columns=['A', 'B'])
    print('\ndf-----------------')
    print(df)

    out_put(df[df['A'] + df['B'] > 10])
    out_put(df.where(df['A'] + df['B'] > 10))
    out_put(df.where(~(df['A'] + df['B'] > 10), 0))
    out_put(df.mask(df['A'] + df['B'] > 10, 0))

    # m = df % 3 == 0
    #     # print(df)
    #     # print('\nwhere-------------------------------------')
    #     # # 条件成立不修改
    #     # print(df.where(m, -df))
    #     # print('\n-------------------------------------')
    #     # # 等价
    #     # print(df.where(m, -df) == pd.np.where(m, df, -df))
    #     # print('\nmask-------------------------------------')
    #     # # 取反
    #     # print(df.where(m, -df) == df.mask(~m, -df))

def get_val1():
    """
    筛选
    :return:
    """
    df = pd.DataFrame([['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1','d1'], ['a2', 'b2', 'c2', 'd2']],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])

    print('\ndf-----------------')
    print(df)
    # 等于某值
    out_put(df[df['a'] == 'a1'])
    # 在列表中
    out_put(df[df['a'].isin(['a1', 'a2'])])
    # 包含某字符串
    out_put(df[df['a'].str.contains('1')])



def get_val():
    """
    获取某个具体的值
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])

    print('\ndf-----------------')
    print(df)
    print('\n取a列00行-----------------')
    out_put(df['a'][0])
    out_put(df['a']['00'])

    out_put(df.loc['00']['a'])
    out_put(df.iloc[0]['a'])

    print('\n取区间-----------------')
    out_put(df['a'][:1])
    out_put(df.loc[:'01', :'b'])
    out_put(df.loc[['00', '01'], ['a', 'b', 'c']])
    out_put(df.iloc[:1, :1])
    out_put(df.iloc[[0, 1], [0, 1, 1]])

    out_put(df[:1][:1])

    out_put(df.iloc[1, 1])
    out_put(df.iat[1, 1])


def get_col():
    """
    获取行
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-----------------')
    print(df)

    print('\n切片 按列名取单列 series-----------------')
    out_put(df['a'])

    print('\n切片 按列名取多列 DataFrame-----------------')
    out_put(df[['a', 'b', 'b']])

    print('\n切片 按列位置取多列 DataFrame-----------------')
    out_put(df.ix[:, [0, 1, 1]])

    pass


def get_row_no_index():
    """
    DataFrame无index
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      columns=['a', 'b', 'c', 'd'])

    print('\ndf-----------------')
    print(df)
    print('\n切片 按索引取多行 前闭后开 得DataFrame-----------------')
    out_put(df[0:1])

    print('\nloc 按索引取单行 得Series-----------------')
    out_put(df.loc[0])

    print('\nloc 按值取多行 前闭后闭 得DataFrame-----------------')
    out_put(df.loc[0:1])
    out_put(df.loc['0':'1'])

    print('\niloc 按索引取单行 得DataFrame-----------------')
    out_put(df.iloc[0])
    print('\niloc 按值取多行 前闭后开 得DataFrame-----------------')
    out_put(df.iloc[0:2])

    # ------------------------------------------------------------
    # 结论
    # ------------------------------------------------------------

    pass


def get_row1():
    """
    读取行
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-----------------')
    print(df)

    # 直接切片
    print('\n切片 按值取多行 前闭后闭 得DataFrame-----------------')
    print(type(df['00':'01']))
    print(df['00':'01'])

    print('\n切片 按索引取多行 前闭后开 得DataFrame-----------------')
    print(type(df[0:1]))
    print(df[0:1])

    # loc 按列值
    print('\nloc 按值取单行 得Series-----------------')
    print(type(df.loc['00']))
    print(df.loc['00'])

    print('\nloc 按值取单行 得DataFrame-----------------')
    print(type(df.loc[['00']]))
    print(df.loc[['00']])

    print('\nloc 按值取非连续多行 得DataFrame-----------------')
    print(type(df.loc[['00', '02', '01']]))
    print(df.loc[['00', '02', '01']])

    print('\nloc 按值取连续多行 前闭后闭 得DataFrame-----------------')
    print(type(df.loc['00':'01']))
    print(df.loc['00':'01'])

    # iloc 按索引

    print('\niloc 按索引取单行  得Series-----------------')
    print(type(df.iloc[0]))
    print(df.iloc[0])

    print('\niloc 按索引取单行  得Series-----------------')
    out_put(df.iloc[-1])

    print('\niloc 按索引取单行 得DataFrame-----------------')
    print(type(df.iloc[[0]]))
    print(df.iloc[[0]])

    print('\niloc 按索引取非连续多行 得DataFrame-----------------')
    print(type(df.iloc[[0, 2, 2]]))
    print(df.iloc[[0, 2, 2]])

    print('\niloc 按索引取连续多行 前闭后开 得DataFrame-----------------')
    print(type(df.iloc[0:2]))
    print(df.iloc[0:2])

    out_put(df.iloc[-1:-2])

    # ------------------------------------------------------------
    # 结论
    # 1、切片[]无法取单行。固定格式[,].不支持行列表，所以
    # 2、loc是依靠值取行。iloc是依靠索引取行。
    # 3、
    #           []              [:]                          [[,,]]
    #                           行区间                        行列表
    #   切片     不支持          为值时，连续的行，前闭后闭       不支持      
    #                           为索引时，连续的行，前闭后开
    #   loc     值，单行         值，连续多行，前闭后闭          值，非连续多行，前闭后闭
    #   iloc    索引，单行       索引，连续多行，前闭后开         索引，非连续多行，前闭后闭
    # ------------------------------------------------------------ 


def out_put(val):
    print(type(val))
    print(val)


main()
