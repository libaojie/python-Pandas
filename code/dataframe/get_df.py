import pandas as pd




def main():
    print('\n---------------------------------------------------------------------------------------------------------')
    # 获取行
    # get_row1()
    get_row_no_index()

    print('\n---------------------------------------------------------------------------------------------------------')
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
    # print(df[['00','01','02']])

    print('\n切片 按索引取多行 前闭后开 得DataFrame-----------------')
    print(type(df[0:1]))
    print(df[0:1])
    # 索引无法取单行
    # print(df[0])
    # 索引无法取单行
    # print(df['00'])

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

    print('\niloc 按索引取单行 得DataFrame-----------------')
    print(type(df.iloc[[0]]))
    print(df.iloc[[0]])

    print('\niloc 按索引取非连续多行 得DataFrame-----------------')
    print(type(df.iloc[[0, 2, 2]]))
    print(df.iloc[[0, 2, 2]])

    print('\niloc 按索引取连续多行 前闭后开 得DataFrame-----------------')
    print(type(df.iloc[0:2]))
    print(df.iloc[0:2])

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