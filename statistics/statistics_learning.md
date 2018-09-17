# 統計学入門 #

## 再確認ためのメモ ##

### Pythonで学ぶ統計学の教科書 ###

#### 第３部 第2章 多変量データ編 ####

1. 共分散
1. 分散共分散行列
1. ピアソンの積率相関係数
1. 相関行列

~~~python
import numpy
import pandas
import scipy
from scipy import stats
cov_data = pandas.DataFrame({
    'x':[18.5, 18.7, 19.1, 19.7, 21.5, 21.7, 21.8, 22.0, 23.4, 23.8],
    'y':[34, 39, 41, 38, 45, 41, 52, 44, 44, 49]
})
x = cov_data["x"]
y = cov_data["y"]
N = len(cov_data)
mu_x = scipy.mean(x)
mu_y = scipy.mean(y)

# 共分散計算 分母N
cov_0 = sum((x - mu_x) * (y - mu_y)) / N
# 共分散計算 分母N-1
cov_1 = sum((x - mu_x) * (y - mu_y)) / (N - 1)

# 共分散計算 分母N
scipy.cov(x, y, ddof = 0) 
# 共分散計算 分母N-1
scipy.cov(x, y, ddof = 1) 

# ピアソンの積率相関係数
sigma_2_x = scipy.var(x, ddof = 1)
sigma_2_y = scipy.var(y, ddof = 1)
rho_1 = cov_1 / scipy.sqrt(sigma_2_x * sigma_2_y)

sigma_2_x_sample = scipy.var(x, ddof = 0)
sigma_2_y_sample = scipy.var(y, ddof = 0)
rho_0 = cov_0 / scipy.sqrt(sigma_2_x_sample * sigma_2_y_sample)

# rho_0 = rho_1 is True

# ピアソンの積率相関係数
scipy.corrcoef(x, y)
~~~


#### 第３部 第3章 matplotlib・seabornによるデータの可視化 ####

##### 1. 折り線 #####

~~~python
import numpy
import pandas
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline

x = numpy.array([0,1,2,3,4,5,6,7,8,9])
y = numpy.array([2,3,4,3,5,4,6,7,4,8])

# スタイル1
pyplot.plot(x, y, color = 'black')
pyplot.title("lineplot matplotlib")
pyplot.xlabel("x")
pyplot.ylabel("y")
# グラフ画像保存
# pyplot.savefig("line")
~~~

~~~python
import numpy
import pandas
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline

x = numpy.array([0,1,2,3,4,5,6,7,8,9])
y = numpy.array([2,3,4,3,5,4,6,7,4,8])

# スタイル2
import seaborn
seaborn.set()
pyplot.plot(x, y, color = 'black')
pyplot.title("lineplot matplotlib")
pyplot.xlabel("x")
pyplot.ylabel("y")
~~~

##### 2. ヒストグラム #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

fish_data = numpy.array([2,3,3,4,4,4,4,5,5,6])
# カーネル密度推定しない
seaborn.distplot(fish_data, bins = 5, color = 'black', kde = False)
# カーネル密度推定する
seaborn.distplot(fish_data, bins = 5, color = 'black')
~~~

##### 3. 2変量ヒストグラム #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

fish_multi = pandas.DataFrame({
    'species':['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    'length':[2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9]
})

#fish_multi.groupby("species").describe()

length_a = fish_multi.query('species == "A"')["length"]
length_b = fish_multi.query('species == "B"')["length"]
seaborn.distplot(length_a, bins = 5, color = 'black', kde = False)
seaborn.distplot(length_b, bins = 5, color = 'gray', kde = False)
~~~

##### 4. 箱ひげ図 #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

fish_multi = pandas.DataFrame({
    'species':['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    'length':[2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9]
})
seaborn.boxplot(x = "species", y = "length", data = fish_multi, color = "gray")
~~~

##### 5. バイオリンプロット #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

fish_multi = pandas.DataFrame({
    'species':['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    'length':[2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9]
})
seaborn.violinplot(x = "species", y = "length", data = fish_multi, color = "gray")
~~~

##### 6. 棒グラフ #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

fish_multi = pandas.DataFrame({
    'species':['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    'length':[2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9]
})
seaborn.barplot(x = "species", y = "length", data = fish_multi, color = "gray")
~~~

##### 7. 散布図 #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

cov_data = pandas.DataFrame({
    'x':[18.5, 18.7, 19.1, 19.7, 21.5, 21.7, 21.8, 22.0, 23.4, 23.8],
    'y':[34, 39, 41, 38, 45, 41, 52, 44, 44, 49]
})
seaborn.jointplot(x = "x", y = "y", data = cov_data, color = "black")
~~~

##### 8. ペアプロット #####

~~~python
import numpy
import scipy
%precision 3
from matplotlib import pyplot
%matplotlib inline
import seaborn

iris = seaborn.load_dataset("iris")
# iris.head(n = 3)
#iris.groupby("species").mean()
seaborn.pairplot(iris, hue = "species", palette = 'gray')
~~~

#### 第３部 第4章 母集団からの標本抽出シミュレーション ####

