# pandas #

## ライブラリインポート ##

~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
~~~

~~~python
# python
# list add
a = [1,2,3]
b = [4,5]
a + b
a.extend(b)

# string to char list
list('abcd')

# dict add
x={"a":[1,2,3],"b":[4,5,6],"c":[7,8,9]}
y={"c":[10,11],"d":[12,13]}
x.update(y)

# combine set
m = {1,2}
n = {4,5,6}
m.union(n)
m | n

# jupyter notebook

# ? をobjに付けると詳細情報確認できる
a = [1,2,3]
?a

# IPython 7.5.0 以後
# %time,%%time,%timeit,%%timeit
# コマンド前に %time を付けるとコマンドの実行時間を計測できる 
n = 100000
%time sum(range(n))
# セルに %%time を付けるとセルの実行時間を計測できる
%%time
n = 100000
sum(range(n))
# %timeit,%%timeit 短いコマンドの時は利用

# %matplotlib inline 生成したクラブをインラインで表示
# %who 宣言した変数一覧表示
%who
%who str
%who function
~~~

~~~python
# Series
height_list=[185,162,171,155,191,166]
height_series=pd.Series(height_list)
print(height_series)

weight_arr=np.array([72,51,69,55,87,78])
weight_series=pd.Series(weight_arr)
print(weight_series)

ser = pd.Series([1,2,3],name='some series')
print(ser)
print(ser.index)

dic = {'T':185,'H':162,'B':171,'R':155,'M':191,'S':166}
ser = pd.Series(dic)
print(ser)

dic={'a':0,'b':1,'c':2}
a = pd.Series(dic,index=['a','b','c','d'])
print(a) # d NaN

pd.Series(10,index=['A','B','C'])

ser = pd.Series([1,2,3,4,5],index=['A','B','C','D','E'])
ser
ser + 1 # Series 数値演算
ser[1:3] # Series 要素取得
ser[ser > 3] # Series 抽出
ser['A'] # Series 抽出
ser.loc['A'] # Series 抽出(列)
ser.iloc[0] # Series 抽出(行)
ser2 = pd.Series([4,5,6,7],index=['D','E','F','G'])
ser + ser2 # Series 演算、ラベルが同一の要素で計算
ser.dtype # Series 要素のtype

pd.Series(['a','b','c']) # numpy配列
pd.Series(['a','b','c'], dtype='category') # numpy配列ではない

ser.append(ser2) # deprecated
pd.concat([ser,ser2])
ser.append(ser2,ignore_index=True) # deprecated
pd.concat([ser,ser2],ignore_index=True)

# Series 重複値
ser=pd.Series([1,1,2,2,2,3,4,5,6,6])
ser.drop_duplicates() # default keep='first'
ser.drop_duplicates(keep='first')
ser.drop_duplicates(keep='last')
ser.drop_duplicates(keep=False)

# Series 欠損値
ser=pd.Series([1,np.nan,np.nan,4,np.nan],index=list('abcde'))
ser.isna()
ser.notna()
ser[ser.isna()]
ser[ser.notna()]
ser.dropna()
~~~

~~~python
# DataFrame
pd.DataFrame(['a',1,0.5]) # Series との違いは Column がある
pd.DataFrame([[1,2,3],[4,5,6]],index=['a','b'],columns=['c','d','e'])
pd.DataFrame([[1,2,3],[4,5]],index=['a','b'],columns=['c','d','e']) # 欠損はNaN
pd.DataFrame([[1,2,3],[4,5]],index=['a','b'],columns=['c','d']) # カラム足りないとエラー
pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})

age = pd.Series([10,12,9], index=['A','B','C'])
sex = pd.Series(['M','F','F'], index=['C','A','D'])
pd.DataFrame({'age':age,'sex':sex})

pd.DataFrame({'age':{'A':10,'B':12,'C':9},'sex':{'C':'M','A':'F','D':'F'}})

df = pd.DataFrame({
    'math':[82,93,77],
    'eng':[77,87,71],
    'chem':[69,91,89]
})
df
df['math'] # 列選択 Serise
df[['math','eng']] # 列選択 DataFrame
df[df['math'] > 80] # 行条件選択

df = pd.DataFrame([[1,2,3],[4,5,6]])
df.index
df.index = ['a','b']
df.columns
df.columns = ['c','d','e']
df.loc['a']
df.loc['a',:]
df.loc['b','e']
df.iloc[0,:]
df.iloc[1,2]
df.shape
df.size
df.iloc[1,1] = 100
df['new1'] = 10 # 列追加、同じ値
df['new2'] = [5,6] # 列追加

df = pd.DataFrame([[1,2,3],[4,5,6]])
df['new1'] = 10
df['new2'] = [5,6]
df.append(pd.Series([7,8,9,10,11],index=[0,1,2,'new1','new2'],name='new3')) # deprecated
pd.concat([df,pd.DataFrame([pd.Series([7,8,9,10,11],index=[0,1,2,'new1','new2'],name='new3')])])

pd.DataFrame(pd.Series([7,8,9])) # 3行1列
pd.DataFrame([pd.Series([7,8,9])]) # 1行3列

df = pd.DataFrame([[1,2,3],[4,5,6]],index=['x','y'], columns=['A','B','C'])
df.drop(labels='x',axis=0) # dfからx行削除結果を戻す。dfは変更しない。
df.drop(labels='x',axis=0, inplace=True) # dfからx行削除
df.drop(labels=['A','C'],axis=1)

df=pd.DataFrame([[1,2,3],[4,5,6],[1,2,3],[3,5,6],[1,2,3]],columns=list('ABC'))
df.duplicated(keep='first')
df[df.duplicated(keep='first')]
df.drop_duplicates(keep='first') # inplace = True でオブジェクトに変更を即反映

df=pd.DataFrame([[1,2,3],[4,5,np.nan],[1,np.nan,np.nan],[3,5,6],[7,8,9]],columns=list('ABC'))
df.isna()
df.notna()
df.dropna(axis=1)
df.dropna(axis=0)
~~~

~~~python
# Index
idx1 = pd.Index([1,7,2,3,5])
idx2 = pd.Index([3,4,1,5,6])
print(idx1)
print(idx2)
print(idx1 & idx2) # deprecated
print(idx1.intersection(idx2))
print(idx1 | idx2) # deprecated
print(idx1.union(idx2))

pd.DataFrame([1,2,3]).index # RangeIndex
pd.DataFrame([1,2,3],index=[0,1,2]).index # Int64Index
idx_date = pd.DatetimeIndex(freq='D',start='2018-12-28',end='2019-01-05') # pandas 1.4.2 でエラー、もしかして印刷ミス?
dates = pd.date_range(freq='D',start='2018-12-28',end='2019-01-05')
df_date = pd.DataFrame([1,2,3,4,5,6,7,8,9],index=dates)
df_date.index # DatetimeIndex
df_date
df_date['2019'] # deprecated
df_date.loc['2019']

df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],index=[0,2,4],columns=['a','b','c'])
df
df1 = df.reindex([0,1,2,3,4])
df1
df1.index=[5,6,7,8,9]
df1
df2 = df1.reindex(columns=['a','b','c','d'])
df3 = df2.reindex([5,6,7,8,9,10],method='ffill') # method:{None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’}

ser = pd.Series([1,2,3],index=pd.Index(['い','あ','う']))
ser
ser.sort_index()

domain = 'https://archive.ics.uci.edu'
path = '/ml/machine-learning-databases/adult/'
file = 'adult.data'
df = pd.read_csv(domain + path + file, header=None, skipinitialspace=True)
df.iloc[0:2]
df.shape
df.head()
df.tail()
df.info() # data type 確認
df[4].describe() # 統計量


~~~


