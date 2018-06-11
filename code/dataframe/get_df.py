import pandas as pd




def main():
    print('\n---------------------------------------------------------------------------------------------------------')
    get_row()
    print('\n---------------------------------------------------------------------------------------------------------')
    pass


def get_row():
    """
    读取行
    :return:
    """
    df = create_df()

    # 直接切片
    print('\n切片 按值取多行 前闭后闭 得DataFrame-----------------')
    print(type(df['00':'01']))
    print(df['00':'01'])

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









def create_df():
    """
     创建df
    :return:
    """
    df = pd.DataFrame([[1, 2, '3', 4], [5, '6', '7', 8], [9.0, 10, '11', 12]],
                      index=['00', '01', '02'],
                      columns=['a', 'b', 'c', 'd'])
    print('\ndf-----------------')
    print(df)

    return df

main()