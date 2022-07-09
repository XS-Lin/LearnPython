# PythonとRのメモ #

## 特殊演算子 ##

~~~py
# 割り算(整数)
13 // 2
# 剰余
11 % 3
# 累乗
2 ** 3
~~~

~~~R
# 割り算(整数)
13 %/% 2
# 剰余
11 %% 3
# 累乗
2 ^ 3
~~~

## データ型とキャスト ##

~~~py
# データ型
type(3)
# キャスト
int(3.5)
~~~

~~~R
# データ型
mode(3)
# キャスト
as.integer(3.5)
~~~

## 代入 ##

~~~py
x = 3
~~~

~~~R
x <- 3
3 -> x
~~~

## リストと配列 ##

* Python
  * リストの1番目の要素のインデックスは0
  * 範囲指定可能
  * 異なるデータ型格納可能

* R
  * のベクトルの1番目の要素のインデックスは1
  * 範囲指定可能、インデックス省略不可
  * 異なるデータ型格納不可(自動データ型変換)
  * 異なるデータ型格納ためのlistという構造はあるが、操作時に二重[]が必要、data[[1]]

## 条件分岐 ##

~~~py
a = 3
if a==3:
    print('a is 3')
else:
    print('a is not 3')
~~~

~~~R
a <- 3
if (a == 3) {
    print('a is 3')
} else {
    print('a is not 3')
}
~~~

## 内包表記 ##

~~~py
a = [45,58,61,72,53,47,69,56,48,61]
[i for i in a if i % 2 == 0]
~~~

~~~R
a <- [45,58,61,72,53,47,69,56,48,61]
a[a %% 2 == 0]
~~~

## 関数 ##

~~~py
def bmi(height,weight):
    result = weight / height / height
    return result
bmi(1.88,75)
~~~

~~~R
bmi <- function(height,weight) {
    result <- weight / height / height
    return (result)
}
~~~

## パケージ名取得 ##

~~~py
import sys
sys.modules['__main__']
sys.modules['__main__'].__file__

import os
os.path.basename(file_path)
file_name.split(".")[0]

instance.__class__.__name__

class.__name__
~~~

~~~r
environmentName(environment(select))
environmentName(environment(plot))
~~~

## Get Help ##

~~~py
my_list = []
help(my_list.append)
~~~

~~~r
?`if`
?"if"       # same
help("if")  # same
~~~

## 特記事項 ##

RにはFactor型がある!!!

**[r base factor](https://stat.ethz.ch/R-manual/R-devel/library/base/html/factor.html)**

## 2次元データ ##

~~~py
import pandas as pd
data = [
    ['山田太郎',90,50],
    ['鈴木花子',80,70],
    ['高橋次郎',75,80],
    ['佐藤三郎',88,65]
]
score = pd.DataFrame(data, columns=['氏名','数学','英語'])
score
score.shape
len(score)
len(score.columns)
score['数学']
score.数学
score[score.数学 > 80]
score[score.数学 > 80][['氏名','数学']]
~~~

~~~R
name <- c('山田太郎','鈴木花子','高橋次郎','佐藤三郎')
math <- c(90,80,75,88)
english <- c(50,70,80,65)
score <- data.frame(氏名=name,数学=math,英語=english)
score
dim(score)
nrow(score)
ncol(score)
score$数学
score[,2]
score[score$数学>80,]
score[score$数学>80,c('氏名','数学')]

library(dplyr)
score %>% filter(数学 > 80) %>% select(氏名,数学)
~~~

## 度数分布表とヒストグラム、平均、中央値、最頻値、分散、標準偏差 ##

* [pandas.DataFrame.isin](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html)
* [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
* [na.omit](https://www.rdocumentation.org/packages/psda/versions/1.4.0/topics/na.omit)
* 階級幅は一般的にスタージェスの公式で決める
  $$ 1+log_2n $$

  ~~~py
  import math
  math.log2(47)+1
  ~~~

  ~~~r
  log2(47)+1
  ~~~

* 標準化
  $$ \frac{x_i - \bar x}{\sigma} $$
* 偏差値
  $$ 50 + \frac{x_i - \bar x}{\sigma} \times 10 $$

~~~py
import pandas as pd
kokusei = pd.read_csv('data/02/c01.csv',encoding='shift_jis')
kokusei.head()

kokusei.tail()

kokusei = kokusei.dropna(subset=['都道府県名'])
h27 = kokusei[kokusei['西暦（年）'] == 2015]
h27_pref = h27[~h27['都道府県コード'].isin(['00','0A','0B'])]
len(h27_pref)

h27_population = pd.Series(h27_pref['人口（総数）'],dtype='int')
pd.value_counts(h27_population // 1000000, sort = False)

h27_population.hist()

h27_population.hist(bins=list(range(0,15000000,1000000)))

h27_population.sum()

# 平均
total = h27_population.sum()
total / len(h27_population)

h27_population.mean()

import numpy as np
np.average(h27_population)

# 中央値
h27_population.median()

# 分散
import numpy as np
data = [58,67,61,80,55,72,69,74]
m = np.average(data)
np.sum([(i - m)**2 for i in data]) / len(data)

np.var(data)

# 標準偏差
np.sqrt(np.var(h27_population))

np.std(h27_population)
~~~

~~~R
kokusei <- read.csv('data/02/c01.csv',fileEncoding='shift_jis',stringAsFactors=FALSE)
head(kokusei)

tail(kokusei)

kokusei <- na.omit(kokusei)
h27 <- kokusei[kokusei$西暦.年. == 2015,]
h27_pref <- h27[
    (h27$都道府県コード != '00') &
    (h27$都道府県コード != '0A') &
    (h27$都道府県コード != '0B') ,
]
nrow(h27_pref)

h27_population <- as.integer(h27_pref$人口.総数.)
table(h27_population %/% 1000000)

hist(h27_population)

hist(h27_population,breaks=seq(0,15000000,1000000))

dist = hist(h27_population,breaks=seq(0,15000000,1000000))
dist

sum(h27_population)

# 平均
total <- sum(h27_population)
total / length(h27_population)

mean(h27_population)

# 中央値
median(h27_population)

# 分散
data <- c(58,67,61,80,55,72,69,74)
m <- mean(data)
sum((data - m)**2) / length(data)

# 不偏分散
var(data)

# 標準偏差
h27_mean = mean(h27_population)
sqrt(sum((h27_population - h27_mean) ** 2) / length(h27_population))
~~~

## グラフを描く ##

~~~py
import matplotlib.pyplot as plt
plt.bar(
    ['A','B','C'],
    [100,200,300]
)

plt.plot(
    [1,2,3,4,5,6],
    [800,600,700,1100,900,1000]
)

plt.plot(
    [1,2,3,4,5,6],
    [800,600,700,1100,900,1000],
    color='red'
)
plt.plot(
    [1,2,3,4,5,6],
    [700,800,600,900,1000,800],
    color='blue'
)

plt.pie(
    [40,30,20,10],
    labels=['A','O','B','AB'],
    startangle=90,
    counterclock=False
)
~~~

~~~r
barplot(
    c(100,200,300),
    names.arg = c('A','B','C')
)

plot(
    c(1,2,3,4,5,6),
    c(800,600,700,1100,900,1000),
    type='l'
)

plot(
    c(1,2,3,4,5,6),
    c(800,600,700,1100,900,1000),
    type='l',
    ylim=c(600,1100),
    col='red'
)
par(new=TRUE)
plot(
    c(1,2,3,4,5,6),
    c(700,800,600,900,1000,800),
    type='l',
    ylim=c(600,1100),
    col='blue'
)

pie(
    c(40,30,20,10),
    labels=c('A','O','B','AB'),
    clockwise=TRUE,
)
~~~

## 散布図、共分散と相関係数、クロス集計、移動平均 ##

~~~py
plt.scatter(
    pd.Series(h27_pref['人口（男）'],dtype='int'),
    pd.Series(h27_pref['人口（女）'],dtype='int'),
)

english = [80,60,90,70]
math = [50,70,40,80]
np.cov(english,math,bias=True)

np.cov(english,math)

np.corrcoef(english,math)

score = pd.DataFrame([[80,50],[60,70],[90,40],[70,80]])
score.corr()
~~~

~~~r
plot(
    h27_pref[,'人口.男.'],
    h27_pref[,'人口.女.']
)

english <- c(80,60,90,70)
math <- c(50,70,40,80)
cov(english,math)

cor(english,math)
~~~

### 多変数 ###

* [r base unclass](https://stat.ethz.ch/R-manual/R-devel/library/base/html/class.html)
* [r base plot](https://stat.ethz.ch/R-manual/R-devel/library/base/html/plot.html)
* [r graphics pairs](https://www.rdocumentation.org/packages/graphics/versions/3.6.2/topics/pairs)
* [r base factor](https://stat.ethz.ch/R-manual/R-devel/library/base/html/factor.html)

~~~py
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

pd.plotting.scatter_matrix(iris_df,figsize=(10,10),c=iris.target)
plt.show()

iris_df.corr()
~~~

~~~r
plot(iris[,c(1:4)],pch=21,bg=c(2:4)[unclass(iris$Species)])

pairs(iris[,c(1:4)],pch=21,bg=c(2:4)[unclass(iris$Species)])

cor(iris[,c(1:4)])
~~~

### 多変数の関係性を視覚化 ###

~~~r
# 散布図行列
pairs(iris, panel = panel.smooth)

# 散布図・相関行列図
library(psych)
psych::pairs.panels(iris)

# 相関行列図（円）
library(corrplot)
corrplot::corrplot(cor(iris[,-5]))

# ネットワーク図
library(qgraph)
qgraph(cor(iris[,-5]),edge.labels=T )

# テーブルプロット
library(tabplot)
tableplot(iris)

# モダンな散布図・相関行列etc
library(ggplot2)
library(GGally)
ggpairs(iris,aes_string(colour="Species", alpha=0.5))
~~~

### クロス集計 ###

* 多肢選択式の解答の場合、クロス集計は避けるべき

~~~py
import pandas as pd

df = pd.DataFrame([
    ['男','A'],['女','B'],['男','A'],['男','AB'],
    ['女','O'],['男','A'],['女','A'],['女','AB'],
    ['男','A'],['女','A'],['女','O'],['男','B'],
],columns=['sex','blood_type'])
pd.crosstab(df['sex'],df['blood_type'])

# 正規化
pd.crosstab(df['sex'],df['blood_type'],normalize=True)
~~~

~~~r
df <- data.frame(
    sex=c(
        '男','女','男','男',
        '女','男','女','女',
        '男','女','女','男'
    ),
    blood_type=c(
        'A','B','A','AB',
        'O','A','A','AB',
        'A','A','O','B'
    )
)
table(df$sex,df$blood_type)

# 正規化
df_table <- table(df$sex,df$blood_type)
prop.table(df_table)
~~~

### 移動平均 ###

~~~py
import pandas as pd
import numpy as py

temperature = pd.read_csv('data/02/temperature.csv')
temperature.head()

ma = np.convolve(temperature['平均気温(℃)'],[1/7] * 7)
ma

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.plot(temperature['平均気温(℃)'],label='original')
ax.plot(ma,label='moving average')
plt.legend()
plt.show()
~~~

~~~r
temperature <- read.csv('data/02/temperature.csv')
ma <- stats::filter(temperature[,2],rep(1/7,7))
plot(as.Date(temperature[,1]),temperature[,2],type='l',ylim=c(0,30))
par(new=T)
plot(as.Date(temperature[,1]),ma,type='l',col='red',ylim=c(0,30))
~~~

* 指数平滑化法
  $$
  \begin{align*}
  \text{予測値} &= \alpha \times \text{前回実績値} + (1 - \alpha) \times \text{前回予測値} \\
  &= \text{前回予測値} + \alpha \times (\text{前回実績値} - \text{前回予測値})
  \end{align*}
  $$

~~~py
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

temperature = pd.read_csv('data/02/temperature.csv')
wma = np.convolve(
    temperature['平均気温(℃)'],
    [(30 - i) / (15 * 31) for i in range(30)]
)
ew = temperature['平均気温(℃)'].ewm(alpha=0.1).mean()

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.plot(temperature['平均気温(℃)'],label='original')
ax.plot(wma,label='weighted moving average')
ax.plot(ew,label='ew')
plt.legend()
plt.show()
~~~

* [figure](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html)
* [add_subplot](https://matplotlib.org/3.5.0/api/figure_api.html#matplotlib.figure.Figure.add_subplot)

## 確率 ##

~~~py
import numpy as np
dice = [1,2,3,4,5,6]
np.random.choice(dice,10)

import matplotlib.pyplot as plt
np.random.seed(0)
rand = np.random.choice(dice,10)

plt.hist(rand)

np.random.normal(0,1,100)

data = [
    [100000, 10000, 0],
    [0.001, 0.009, 0.99]
]
sum([x_k * p_k for x_k, p_k in zip(data[0],data[1])])

dice = [
    [1, 2, 3, 4, 5, 6],
    [1/6,1/6,1/6,1/6,1/6,1/6]
]
mean = sum([x_k * p_k for x_k, p_k in zip(dice[0],dice[1])])
sum([(x_k - mean) ** 2 * p_k for [x_k, p_k] in zip(dice[0],dice[1])])

def f(x):
    if x < 0:
        return 0
    elif x < 1:
        return x
    elif x < 2:
        return - x + 2
    else:
        return 0

from scipy import integrate

integrate.quad(f,0.5,1.5)

integrate.quad(f,-np.inf,np.inf)

def F(x):
    return integrate.quad(f, -np.int, x)[0]
F(1.5) - F(0.5)

def calc(x):
    return x * f(x)
integrate.quad(calc, -np.inf, np.inf)

ex = integrate.quad(calc, -np.inf, np.inf)[0]
def diff(x):
    return (x - ex) ** 2 * f(x)
integrate.quad(diff, -np.inf, np.inf)

from scipy.stats import norm
norm.cdf(1.64) # cumulative distribution function 分布関数
norm.ppf(0.95) # percent point function パーセント点関数
norm.pdf(1.64) # probability density function 確率密度関数
~~~

~~~r
as.integer(runif(10,min=1,max=7))

set.seed(0)
rand = as.integer(runif(10,min=1,max=7))
hist(rand,breaks=seq(1,7,0.6))

rnorm(100)

data <- data.frame(
    x = c(100000, 10000, 0),
    p = c(0.001, 0.009, 0.99)
)
sum(data$x * data$p)

dice <- data.frame(
    x = c(1, 2, 3, 4, 5, 6),
    p = c(1/6,1/6,1/6,1/6,1/6,1/6)
)
mean = sum(dice$x * dice$p)
sum((dice$x - mean) ** 2 * dice$p)

# if {...} else if {...} else if {...} else {...} の可読性が高いように見えるが、ベクトル処理できない※単一パラメータの処理は可能
f <- function(x) {
    ifelse(x < 0,
           0,
           ifelse(x < 1,
                  x,
                  ifelse(x < 2,
                         -x + 2,
                         0
                  )
           )
    )
}
f(c(0, 0.5, 1, 1.5, 2))

integrate(f,0.5,1.5)

integrate(f,-Inf,Inf)

F <- function(x) {
    integrate(f, -Inf, x)$value
}
F(1.5) - F(0.5)

calc <- function(x) {
    x * f(x)
}
integrate(calc, -Inf, Inf)$value

ex <- integrate(calc, -Inf, Inf)$value
diff <- function (x) {
    (x - ex) ** 2 * f(x)
}
integrate(diff, -Inf, Inf)$value

pnorm(1.64) # 分布関数
qnorm(0.95) # パーセント点関数
dnorm(1.64) # 確率密度関数
~~~

### 空間推定 ###

* 母分散既知

~~~py
import numpy as np
dice = [1,2,3,4,5,6]
population = np.random.choice(dice, 1000)
var = np.var(population)

from scipy.stats import norm

n=20

sample = np.random.choice(population,n)
m = sample.mean()
z = norm.ppf(0.95)
print(m - z * np.sqrt(var / n))
print(m + z * np.sqrt(var / n))
~~~

~~~r
dice <- c(1,2,3,4,5,6)
population <- as.integer(runif(1000,min=1,max=7))
var <- var(population) # 注意:var関数は不偏分散

n=20
sample <- population[as.integer(runif(n,min=1,max=1001))]
m = mean(sample)
z = qnorm(0.95)
print(m - z * sqrt(var / n))
print(m + z * sqrt(var / n))
~~~

* 母分散未知

~~~py
import numpy as np
from scipy.stats import t

dice = [1,2,3,4,5,6]
n=20

sample = np.random.choice(dice,n)
m = np.mean(sample)
var = np.var(sample,ddof=1)
z = t.ppf(0.95, n - 1)
print(m - z * np.sqrt(var / n))
print(m + z * np.sqrt(var / n))
~~~

~~~r
n <- 20

sample <- as.integer(runif(n, min=1, max=7))
m = mean(sample)
var = var(sample)
z = qt(0.95,n-1)
print(m - z * sqrt(var / n))
print(m + z * sqrt(var / n))
~~~

* ベイズ推定

~~~py
from sklearn import linear_model
import numpy as np
dist = [
    [0]*190 + [1]*30,
    [0]*10 + [1]*20
]
sample = [
    np.random.choice(dist[0],20),
    np.random.choice(dist[1],20)
]
mail = [0,1] # 0:通常のメール 1:迷惑メール
reg = linear_model.BayesianRidge()
reg.fit(sample,mail)

reg.predict([np.random.choice(dist[0],20)])
reg.predict([np.random.choice(dist[1],20)])
~~~

## 帰無仮説、対立仮説、有意水準 ##

* 検定手順
  1. 仮説の設定(帰無仮説、対立仮説:片側仮説・両側仮説)
  1. 検定統計量の選定
  1. 有意水準・棄却域の決定(一般的に0.05、厳しい時0.01)
  1. 帰無仮説の棄却または採択(棄却域に入る場合帰無仮説は棄却され対立仮説が採択される)

* 母分散が分かる場合

~~~py
from scipy.stats import norm
[norm.ppf(0.025),norm.ppf(0.975)]
import numpy as np
data = [29.2,29.8,31.2,32.1,28.8,30.1,30.9,29.4,30.7,31.2]
m = np.mean(data)
m

n = len(data)
z_value = (m - 30.0) / np.sqrt(1.5 / n)
z_value

p_value = norm.pdf(z_value)
print(p_value)

norm.cdf(z_value)

# 図で確認
from matplotlib import pyplot as plt

x = np.linspace(-4,4,1000)
fig,ax = plt.subplots(1,1)

ax.plot(x, norm.pdf(x), linestyle='-', label='n='+str(n))

ax.plot(
    z_value,
    p_value, 
    'x', 
    color='red', 
    markersize=7,
    markeredgewidth=2,
    alpha=0.8,
    label='data'
    )

bottom, up = norm.interval(alpha=0.95,loc=0,scale=1)

plt.fill_between(
    x,
    norm.pdf(x),
    0,
    where=(x>=up)|(x<=bottom),
    facecolor='black',
    alpha=0.1
)

plt.xlim(-4,4)
plt.ylim(0,0.4)

plt.legend()
plt.show()
~~~

~~~r
c(qnorm(0.025),qnorm(0.975))
data <- c(29.2,29.8,31.2,32.1,28.8,30.1,30.9,29.4,30.7,31.2)
m <- mean(data)
m

n <- length(data)
z_value <- (m - 30.0) / sqrt(1.5 / n)
z_value

p_value <- dnorm(z_value)
p_value

pnorm(z_value)


~~~

* 母分散がわからない場合

~~~py
from scipy.stats import t

data = [25.1,23.9,25.2,24.6,24.3,24.8,23.8]
n = len(data)
[t.ppf(0.025,n-1),t.ppf(0.975,n-1)]

import numpy as np

m = np.mean(data)
var = np.var(data,ddof=1)
[m,var]

t_value = (m - 25.0) / np.sqrt(var / n)
t_value

p_value = t.pdf(t_value, n - 1)
p_value

t.cdf(t_value,n-1)

# 図で確認
from matplotlib import pyplot as plt

x = np.linspace(-4,4,1000)
fig,ax = plt.subplots(1,1)

ax.plot(x, t.pdf(x, n-1),linestyle='-',label='n='+str(n))
ax.plot(t_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')

bottom,up = t.interval(0.95,n-1)
plt.fill_between(x,t.pdf(x,n-1),0,where=(x>=up)|(x<=bottom),facecolor='black',alpha=0.1)

plt.xlim(-4,4)
plt.ylim(0,0.4)

plt.legend()
plt.show()

# t検定関数利用
from scipy import stats
data = [25.1,23.9,25.2,24.6,24.3,24.8,23.8]
stats.ttest_1samp(data,25.0) # pvalueは両側検定のp値
~~~

~~~r
data <- c(25.1,23.9,25.2,24.6,24.3,24.8,23.8)
n <- length(data)
c(qt(0.025,n-1),qt(0.975,n-1))

m <- mean(data)
var <- var(data)
c(m,var)

t_value <- (m - 25.0) / sqrt(var / n)
t_value

p_value <- dt(t_value, n - 1)
p_value

pt(t_value, n - 1)

# t検定関数利用
data <- c(25.1,23.9,25.2,24.6,24.3,24.8,23.8)
t.test(data,mu=25,conf.level=0.95)　# p-valueは両側検定のp値
~~~

* 母集団が2つ(対応のあるデータ)

~~~py
from scipy.stats import t
data = [2,11,-2,2,4,4,4,8]
n=len(data)
t.ppf(0.95,n-1)
import numpy as np
m = np.mean(data)
var = np.var(data,ddof=1)
[m,var]
t_value = m / np.sqrt(var / n)
t_value
p_value = t.pdf(t_value,n-1)
p_value
t.cdf(t_value,n-1)

# 確認
from matplotlib import pyplot as plt
x = np.linspace(-4,4,1000)
fig,ax = plt.subplots(1,1)
ax.plot(x,t.pdf(x,n-1),linestyle='-',label='n='+str(n))
ax.plot(t_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')
bottom,up = t.interval(0.90,n-1)
plt.fill_between(x,t.pdf(x,n-1),0,where=(x>=up),facecolor='black',alpha=0.1)
plt.xlim(-4,4)
plt.ylim(0,0.4)
plt.legend()
plt.show()

# 関数利用
from scipy import stats
before = [80,75,63,88,91,58,67,72]
after = [82,86,61,90,95,62,71,80]
stats.ttest_rel(before,after)
~~~

~~~r
data <- c(2,11,-2,2,4,4,4,8)
n <- length(data)
qt(0.95,n-1)
m <- mean(data)
var <- var(data)
c(m,var)
t_value <- m / sqrt(var / n)
t_value
p_value <- dt(t_value,n-1)
p_value
pt(t_value,n-1)

# 関数利用
before <- c(80,75,63,88,91,58,67,72)
after <- c(82,86,61,90,95,62,71,80)
t.test(x=before,y=after,paired=T)
~~~

* 母集団が2つ(対応のないデータ)

||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|
|畑1|15|18|17|18|19|17|16|20|19|14|
|畑2|17|20|15|18|16|15|17|18|||

~~~py
from scipy.stats import t
x_1 = [15,18,17,18,19,17,16,20,19,14]
x_2 = [17,20,15,18,16,15,17,18]
n_1 = len(x_1)
n_2 = len(x_2)
t.ppf(0.95,n_1+n_2-2)

import numpy as np
m_1 = np.mean(x_1)
m_2 = np.mean(x_2)
s_1 = np.var(x_1,ddof=1)
s_2 = np.var(x_2,ddof=1)
[m_1,m_2,s_1,s_2]

t_value = (m_1 - m_2) / np.sqrt(
    (1/n_1 + 1/n_2) * 
    ((n_1-1)*s_1 + (n_2 - 1)*s_2) /
    (n_1+n_2-2)
)
t_value

p_value = t.pdf(t_value,n_1+n_2-2)
p_value

t.cdf(t_value,n_1+n_2-2)

# 関数利用
from scipy import stats
x_1 = [15,18,17,18,19,17,16,20,19,14]
x_2 = [17,20,15,18,16,15,17,18]
stats.ttest_ind(x_1,x_2)
~~~

~~~r
x_1 <- c(15,18,17,18,19,17,16,20,19,14)
x_2 <- c(17,20,15,18,16,15,17,18)
n_1 <- length(x_1)
n_2 <- length(x_2)
qt(0.95,n_1+n_2-2)

m_1 <- mean(x_1)
m_2 <- mean(x_2)
s_1 <- var(x_1)
s_2 <- var(x_2)
c(m_1,m_2,s_1,s_2)

t_value <- (m_1 - m_2) / sqrt(
    (1/n_1 + 1/n_2) * 
    ((n_1-1)*s_1 + (n_2 - 1)*s_2) /
    (n_1+n_2-2)
)
t_value

p_value <- dt(t_value,n_1+n_2-2)
p_value

pt(t_value,n_1+n_2-2)

# 関数利用
x_1 <- c(15,18,17,18,19,17,16,20,19,14)
x_2 <- c(17,20,15,18,16,15,17,18)
t.test(x_1,x_2,var.equal=T)
~~~

* 分散を検定

~~~py
from scipy.stats import chi2
chi2.interval(0.90,9)

import numpy as np
data = [31,42,29,51,45,42,37,48,39,50]
n = len(data)
chi_value = (n-1)*np.var(data,ddof=1) / 30
chi_value

# イメージ
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2
p_value = chi2.pdf(chi_value,n-1)
x = np.linspace(0,20,1000)
fig,ax = plt.subplots(1,1)
ax.plot(x,chi2.pdf(x,n-1),linestyle='-',label='n='+str(n-1))
ax.plot(chi_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')
bottom,up = chi2.interval(0.90,n-1)
plt.fill_between(x,chi2.pdf(x,n-1),0,where=(x>=up)|(x<=bottom),facecolor='black',alpha=0.1)
plt.xlim(0,20)
plt.ylim(0,0.12)
plt.legend()
plt.show()
~~~

~~~r
c(qchisq(0.05,9),qchisq(0.95,9))

data <- c(31,42,29,51,45,42,37,48,39,50)
n <- length(data)
chi_value <- (n-1)*var(data)/30
chi_value
~~~

~~~py
from scipy.stats import f
a = [1400,1800,1100,2500,1300,2200,1900,1600]
b = [1500,1800,2200,1200,2000,1700]
n_1 = len(a)
n_2 = len(b)
[f.ppf(0.05,n_1-1,n_2-1),f.ppf(0.95,n_1-1,n_2-1)]

import numpy as np
s_a = np.var(a,ddof=1)
s_b = np.var(b,ddof=1)
f_value = s_a / s_b
f_value

from matplotlib import pyplot as plt
import numpy as np
p_value = f.pdf(f_value,n_1-1,n_2-1)
x = np.linspace(0,6,1000)
fig,ax = plt.subplots(1,1)
ax.plot(x,f.pdf(x,n_1-1,n_2-1),linestyle='-',label='F')
ax.plot(f_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')
bottom,up = f.interval(0.90,n_1-1,n_2-1)
plt.fill_between(x,f.pdf(x,n_1-1,n_2-1),0,where=(x>=up)|(x<=bottom),facecolor='black',alpha=0.1)
plt.xlim(0,6)
plt.ylim(0,1)
plt.legend()
plt.show()
~~~

~~~r
a <- c(1400,1800,1100,2500,1300,2200,1900,1600)
b <- c(1500,1800,2200,1200,2000,1700)
n_1 <- length(a)
n_2 <- length(b)
c(qf(0.05,n_1-1,n_2-1),qf(0.95,n_1-1,n_2-1))

# 関数利用
var.test(a,b)
~~~

* クロス集計

| No | リンク元 | 購入有無 |
| --- | --- | --- |
|  1 | メール | 購入した |
|  2 | メール | 購入しなかった |
|  3 | 広告　 | 購入した |
|  4 | メール | 購入しなかった |
|  5 | 広告　 | 購入しなかった |
|  6 | メール | 購入した |
|  7 | 広告　 | 購入した |
|  8 | メール | 購入しなかった |
|  9 | メール | 購入した |
| 10 | メール | 購入しなかった |
| 11 | 広告　 | 購入した |
| 12 | メール | 購入しなかった |
| 13 | 広告　 | 購入しなかった |
| 14 | メール | 購入した |
| 15 | 広告　 | 購入した |
| 16 | メール | 購入しなかった |
| 17 | メール | 購入した |
| 18 | 広告　 | 購入した |
| 19 | メール | 購入しなかった |
| 20 | 広告　 | 購入した |

~~~py
from scipy.stats import chi2
chi2.interval(0.95,1)

import scipy.stats as st
data = [[5,7],[6,2]]
st.chi2_contingency(data,correction=False)
~~~

~~~r
c(qchisq(0.025,1),qchisq(0.975,1))

m <- matrix(c(5,7,6,2),nrow=2,byrow=F)
chisq.test(m,correct=F)
~~~

* 比率の検定

| 曜日　 | 日 | 月 | 火 | 水 | 木 | 金 | 土 | 計 |
|---|---|---|---|---|---|---|---|---|
| 売上高 | 31 | 16 | 14 | 15 | 13 | 22 | 29 | 140 |

~~~py
from scipy.stats import chi2
chi2.interval(0.90,6)

import numpy as np
data = [31,16,14,15,13,22,29]
m = np.mean(data)
chi_value = sum([(i-m) ** 2 / m for i in data])
chi_value

# イメージ
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2

n = len(data)
p_value = chi2.pdf(chi_value,n-1)

x = np.linspace(0,20,1000)
fig,ax = plt.subplots(1,1)
ax.plot(x,chi2.pdf(x,n-1),linestyle='-',label='n='+str(n-1))
ax.plot(chi_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')
bottom,up = chi2.interval(0.90,n-1)
plt.fill_between(x,chi2.pdf(x,n-1),0,where=(x>=up)|(x<=bottom),
                facecolor='black',alpha=0.1)
plt.xlim(0,20)
plt.ylim(0,0.15)

plt.legend()
plt.show()

# 関数利用
from scipy import stats
stats.chisquare([31,16,14,15,13,22,29])

~~~

~~~r
c(qchisq(0.05,6),qchisq(0.95,6))

data <- c(31,16,14,15,13,22,29)
m <- mean(data)
chi_value <- sum((data - m) ** 2 / m)
chi_value

# 関数利用
chisq.test(c(31,16,14,15,13,22,29))
~~~

| 曜日　　　 | 日 | 月 | 火 | 水 | 木 | 金 | 土 | 計 |
|-----------|---|---|---|---|---|---|---|---|
| 売上高　　 | 31 | 16 | 14 | 15 | 13 | 22 | 29 | 140 |
| 予想売上高 | 30 | 15 | 15 | 15 | 15 | 20 | 40 | 150 |

~~~py
import numpy as np
data = [31,16,14,15,13,22,29]
expected = [30,15,15,15,15,20,40]
chi_value = sum([(i-j) ** 2 / j for i,j in zip(data,expected)])
chi_value

# イメージ
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2

n = len(data)
p_value = chi2.pdf(chi_value,n-1)

x = np.linspace(0,20,1000)
fig,ax = plt.subplots(1,1)
ax.plot(x,chi2.pdf(x,n-1),linestyle='-',label='n='+str(n-1))
ax.plot(chi_value,p_value,'x',color='red',markersize=7,markeredgewidth=2,alpha=0.8,label='data')
bottom,up = chi2.interval(0.90,n-1)
plt.fill_between(x,chi2.pdf(x,n-1),0,where=(x>=up)|(x<=bottom),
                facecolor='black',alpha=0.1)
plt.xlim(0,20)
plt.ylim(0,0.15)

plt.legend()
plt.show()

# 関数利用
from scipy import stats
stats.chisquare(
    [31,16,14,15,13,22,29],
    [30,15,15,15,15,20,40]
)
# 実現値の和と予測値の和に違いがあるため、エラーメッセージが表示される
~~~

~~~r
data <- c(31,16,14,15,13,22,29)
expected <- c(30,15,15,15,15,20,40)
chi_value <- sum((data - expected) ** 2 / expected)
chi_value

# 関数利用
chisq.test(
    c(31,16,14,15,13,22,29),
    p=c(30,15,15,15,15,20,40)/150
)
~~~

* ウェルチの検定、分散

$$ t = \frac{\bar x_1 - \bar x_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}  $$

$$ \nu =
    \frac{
        \Bigl(
            \frac{s_1^2}{n_1} +
            \frac{s_2^2}{n_2}
        \Bigl)^2
    }
    {
        \frac{
            \Bigl(
                \frac{s_1^2}{n_1}
            \Bigl)^2
        }
        {
            n_1 - 1
        } +
        \frac{
            \Bigl(
                \frac{s_2^2}{n_2}
            \Bigl)^2
        }
        {
            n_2 - 1
        }
    }
$$

* Latexメモ

$$ ( \bigl( \Bigl( \biggl( \Biggl( $$
$$ ( \bigm( \Bigm( \biggm( \Biggm( $$
$$ ( \bigr( \Bigr( \biggr( \Biggr( $$

$$
\LaTeX
\newline
\begin{align}
E &=& mc^2                              \\
m &=& \frac{m_0}{\sqrt{1-\frac{v^2}{c^2}}}
\end{align}
$$

||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|
|畑1|15|18|17|18|19|17|16|20|19|14|
|畑2|17|20|15|18|16|15|17|18|||

~~~py
import numpy as np
x_1 = [15,18,17,18,19,17,16,20,19,14]
x_2 = [17,20,15,18,16,15,17,18]
s_1 = np.var(x_1,ddof=1)
s_2 = np.var(x_2,ddof=1)
n_1 = len(x_1)
n_2 = len(x_2)
nu = (s_1 / n_1 + s_2 / n_2) ** 2 / ((s_1 / n_1) ** 2 / (n_1 - 1) + (s_2 / n_2) ** 2 / (n_2 - 1))
nu

from scipy.stats import t
t.ppf(0.95,16)

from scipy.stats import t
t.ppf(0.95,16)
m_1 = np.mean(x_1)
m_2 = np.mean(x_2)
t = (m_1 - m_2) / np.sqrt(s_1 / n_1 + s_2 / n_2)
t

# 関数利用
from scipy import stats
x_1 = [15,18,17,18,19,17,16,20,19,14]
x_2 = [17,20,15,18,16,15,17,18]
stats.ttest_ind(x_1,x_2,equal_var = False)
~~~

~~~r
x_1 <- c(15,18,17,18,19,17,16,20,19,14)
x_2 <- c(17,20,15,18,16,15,17,18)
s_1 <- var(x_1)
s_2 <- var(x_2)
n_1 <- length(x_1)
n_2 <- length(x_2)
nu <- (s_1 / n_1 + s_2 / n_2) ** 2 / ((s_1 / n_1) ** 2 / (n_1 - 1) + (s_2 / n_2) ** 2 / (n_2 - 1))
nu

qt(0.95,16)

m_1 <- mean(x_1)
m_2 <- mean(x_2)
t <- (m_1 - m_2) / sqrt(s_1 / n_1 + s_2 / n_2)
t

# 関数利用
t.test(x_1,x_2)
~~~

|店舗|A|B|C|
|---|---|---|---|
|売上|36|42|51|
|　　|40|40|58|
|　　|38|48|56|
|　　|42|35|52|
|　　|45|37|49|
|　　|43|38|50|
|　　|39|43|51|

~~~py
from scipy import stats
a = [36,40,38,42,45,43,39]
b = [42,40,48,35,37,38,43]
c = [51,58,56,52,49,50,51]
stats.f_oneway(a,b,c)
~~~

## 回帰分析 ##

|日付|気温|販売数|
|||||
|:---:|:---:-:---:-:---:-:|---:|
|7/1|15|80|
|7/2|21|100|
|7/3|22|95|
|7/4|24|120|
|7/5|25|128|
|7/6|27|140|
|7/7|28|141|
|7/8|29|150|
|7/9|30|160|

~~~py
from sklearn.linear_model import LinearRegression
x = [[15],[21],[22],[24],[25],[27],[28],[29],[30]]
y = [80,100,95,120,128,140,141,150,160]
model_lr = LinearRegression()
model_lr.fit(x,y)
print('y = %.03fx + %.03f' % (model_lr.coef_,model_lr.intercept_))

import matplotlib.pyplot as plt
plt.plot(x,y,'o')
plt.plot(x,model_lr.predict(x),linestyle='solid')
plt.show()

model_lr.score(x,y) # 実務上、決定係数が0.5以上であれば使える
~~~

~~~r
x <- c(15,21,22,24,25,27,28,29,30)
y <- c(80,100,95,120,128,140,141,150,160)
lm(y ~ x)

x <- c(15,21,22,24,25,27,28,29,30)
y <- c(80,100,95,120,128,140,141,150,160)
z <- lm(y ~ x)
plot(x,y)
abline(z)

summary(lm(y ~ x)) # 決定係数 Mutiple R-squared
~~~

* 回帰式の係数

$$
\begin{align*}
a &= \frac{
    \frac{1}{n}\sum_{k=1}^{n}(x_k - \bar x)(y_k - \bar y)
}{
    \frac{1}{n}\sum_{k=1}^{n}(x_k - \bar x)^2
}
\newline
&= \frac{S_{xy}}{S_{xx}}
\end{align*}
$$

$$ b = \bar y - a \bar x $$

~~~py
x = [15,21,22,24,25,27,28,29,30]
y = [80,100,95,120,128,140,141,150,160]
x_mean = sum(x)/len(x)
y_mean = sum(y)/len(y)
sum_xy = sum([(i - x_mean) * (j - y_mean) for (i,j) in zip(x,y)])
sum_xx = sum([(i - x_mean) ** 2 for i in x])
a = sum_xy / sum_xx
b = y_mean - a * x_mean
[a,b]
~~~

~~~r
x <- c(15,21,22,24,25,27,28,29,30)
y <- c(80,100,95,120,128,140,141,150,160)
x_mean <- mean(x)
y_mean <- mean(y)
sum_xy <- sum((x - x_mean) * (y - y_mean))
sum_xx <- sum((x - x_mean) ** 2)
a <- sum_xy / sum_xx
b <- y_mean - a * x_mean
print(c(a,b))
~~~

* 決定係数

$$ R^2 = \frac{\sum_{k=1}^{n}(\hat y_k - \bar y)^2}{\sum_{k=1}^{n}(y_k - \bar y)^2} $$

~~~py
def f(x):
    return 5.572 * x + -13.054
x = [15,21,22,24,25,27,28,29,30]
y = [80,100,95,120,128,140,141,150,160]
mean_y = sum(y)/len(y)
sum_data = sum([(i - mean_y) ** 2 for i in y])
sum_predict = sum([(f(i) - mean_y) ** 2 for i in x])
sum_predict / sum_data
~~~

~~~r
f <- function (x) {
    return (5.572 * x + -13.054)
}
x <- c(15,21,22,24,25,27,28,29,30)
y <- c(80,100,95,120,128,140,141,150,160)
mean_y <- mean(y)
sum_data <- sum((y - mean_y) ** 2)
sum_predict <- sum((f(x) - mean_y) ** 2)
sum_predict / sum_data
~~~

## 重回帰分析 ##

| No | 走行距離(km) | 経過年数(年) | 金額(万円) |
|:---:|---:|:---:|---:|
| 1 | 31,438 | 3 | 125 |
| 2 | 13,845 | 3 | 140 |
| 3 | 43,095 | 5 | 98 |
| 4 | 40,946 | 4 | 113 |
| 5 | 82,375 | 5 | 65 |
| 6 | 78,764 | 6 | 70 |
| 7 | 90,554 | 7 | 55 |
| 8 | 69,142 | 8 | 80 |
| 9 | 23,712 | 5 | 95 |
| 10 | 51,489 | 6 | 88 |
| 11 | 60,023 | 10 | 90 |
| 12 | 80,123 | 2 | 132 |

* 決定係数
$$ R^2 = \frac{\sum_{k=1}^{n}(\hat y_k - \bar y)^2}{\sum_{k=1}^{n}(y_k - \bar y)^2} $$

* 自由度調整済み決定係数

$$ R_{adj}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1} $$

~~~py
from sklearn.linear_model import LinearRegression
data = [
    [31438,3],[13845,3],[43095,5],[40946,4],
    [82375,5],[78764,6],[90554,7],[69142,8],
    [23712,5],[51489,6],[60023,10],[80123,2]
]
price = [125,140,98,113,65,70,55,80,95,88,90,132]
model_lr = LinearRegression()
model_lr.fit(data,price)
print([model_lr.coef_,model_lr.intercept_])

r = model_lr.score(data,price)
1 - (1 - r) * (len(price) - 1) / (len(price) - len(data[0]) - 1)
~~~

~~~r
distance <- c(
    31438,13845,43095,40946,
    82375,78764,90554,69142,
    23712,51489,60023,80123
)
year <- c(3,3,5,4,5,6,7,8,5,6,10,2)
price <- c(125,140,98,113,65,70,55,80,95,88,90,132)
lm(price ~ distance + year)

summary(lm(price ~ distance + year))
~~~

**注意**
誤差の分布は正規分布であることを仮定している。つまり、直線の両側に集まっているような分布のこと。仮定できなければ、一般化線形モデルなどを使います。

## 数量化理論I類 ##

* 目的変数のある場合の分析手法
* 説明変数のデータ形態がカテゴリーデータである
* 目的変数と説明変数の関係を調べて次の3項目を明らかにする
  1. 説明変数の、目的変数に対する貢献度
  1. 説明変数の重要度ランキング
  1. 予測

## ロジスティック回帰 ##

$$ (\text{故障率}) = a \times (\text{室温}) + b \times (\text{湿度})  + c \times (\text{使用頻度}) + d $$

|故障有無(1:有、0:無)|室温(℃)|湿度(％)|使用頻度(回/日)|
|:---:|:---:|:---:|:---:|
| 1 | 25.0 | 80 | 5 |
| 1 | 27.1 | 65 | 3 |
| 1 | 28.2 | 64 | 6 |
| 1 | 32.3 | 72 | 4 |
| 1 | 33.8 | 82 | 4 |
| 0 | 25.3 | 45 | 2 |
| 0 | 24.7 | 52 | 1 |
| 0 | 26.3 | 60 | 3 |
| 0 | 28.2 | 70 | 1 |
| 0 | 27.6 | 49 | 4 |

~~~py
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.DataFrame([
    [ 1 , 25.0 , 80 , 5 ],
    [ 1 , 27.1 , 65 , 3 ],
    [ 1 , 28.2 , 64 , 6 ],
    [ 1 , 32.3 , 72 , 4 ],
    [ 1 , 33.8 , 82 , 4 ],
    [ 0 , 25.3 , 45 , 2 ],
    [ 0 , 24.7 , 52 , 1 ],
    [ 0 , 26.3 , 60 , 3 ],
    [ 0 , 28.2 , 70 , 1 ],
    [ 0 , 27.6 , 49 , 4 ]
])
df.columns = ['malfunction','temprature','humidity','frequency']
x_train = df[['temprature','humidity','frequency']]
y_train = df['malfunction']
model = LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
model.score(x_train,y_train)

[model.coef_,model.intercept_]
~~~

~~~r
data <- data.frame(
    malfunction = c(1,1,1,1,1,0,0,0,0,0),
    temperature = c(25.0,27.1,28.2,32.3,33.8,25.3,24.7,26.3,28.2,27.6),
    humidity = c(80,65,64,72,82,45,52,60,70,49),
    frequency = c(5,3,6,4,4,2,1,3,1,4)
)
model <- glm(malfunction ~ temperature + humidity + frequency, family=binomial,data=data)
summary(model)
~~~
