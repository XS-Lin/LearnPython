# 統計学入門 #

## 数式表記（TeX Math） ##

[利用可能な数式](https://katex.org/docs/supported.html)
[(アルファベット順)](https://katex.org/docs/support_table.html)

$$ a' a'' a''' a^{\prime} \acute{a} \bar{a} \breve{a} \check{a} $$

$$ \dot{a} \ddot{a} \grave{a} \hat{\theta} \widehat{a+b} \tilde{a} $$

$$ \widetilde{AB} \utilde{AB} \vec{F} \overleftarrow{AB} $$

$$ \underleftarrow{AB} \overleftharpoon{ac} \overleftrightarrow{AB} $$

$$ \underleftrightarrow{AB} \overline{AB} \underline{AB} \widecheck{ac} $$

$$ \mathring{g} \overgroup{AB} \undergroup{AB} \Overrightarrow{AB} $$

$$ \overrightarrow{AB} \underrightarrow{AB} \overrightharpoon{AB} $$

$$ \overbrace{AB} \underbrace{AB} \overlinesegment{AB}  $$

$$ \underlinesegment{AB} $$

$$ () [] \{ \} < > | \| $$

$$ \lvert \rvert \lang \rang \lt \gt \le \ge \lbrack \rbrack $$

$$ \lbrace \rbrace \langle \rangle \vert \Vert \lVert \rVert $$

$$ \lceil \rceil \lfloor \rfloor \lmoustache \rmoustache \lgroup \rgroup $$

$$ \ulcorner \urcorner \left. \right. \backslash $$

$$ \uparrow \downarrow \leftarrow \rightarrow $$

$$ \Uparrow \Downarrow \Leftarrow \Rightarrow $$

$$ \left(\LARGE{AB}\right) $$

$$ ( \big( \Big( \bigg( \Bigg( $$

~~~math
\left \middle \right
\big \Big \bigg \Bigg
\bigl \Bigl \biggl \Biggl
\bigm \Bigm \biggm \Biggm
\bigr \Bigr \biggr \Biggr
~~~

$$
\begin{matrix}
a & b \\
c & d 
\end{matrix}
$$

$$
\begin{array}{cc}
a & b \\
c & d
\end{array}
$$

$$ \begin{pmatrix} a & b \\ c & d \end{pmatrix} $$

$$ \begin{bmatrix} a & b \\ c & d \end{bmatrix} $$

$$ \begin{vmatrix} a & b \\ c & d \end{vmatrix} $$

$$ \begin{Vmatrix} a & b \\ c & d \end{Vmatrix} $$

$$ \begin{Bmatrix} a & b \\ c & d \end{Bmatrix} $$

$$ \def\arraystrech{1.5}
   \begin{array}{c:c:c}
   a & b & c \\ \hline
   d & e & f \\ \hdashline
   g & h & i
   \end{array}
$$

$$
\begin{aligned}
a &= b + c \\
d + e &= f
\end{aligned}
$$

$$
\begin{alignedat}{2}
10&x + &3&y = 2 \\
3&x + &13&y =4
\end{alignedat} 
$$

$$ \begin{gathered} a=b \\ e=b+c \end{gathered} $$

$$ x = \begin{cases} 
       a &\text{if } b \\
       c &\text{if } d
       \end{cases}
$$

$$ \Alpha \Beta \Gamma \Delta \Epsilon \Zeta \Eta \Theta \Iota \Kappa \Lambda \Mu \Nu \Xi \Omicron \Pi \Sigma \Tau \Upsilon \Phi \Chi \Psi \Omega  $$
$$ \varGamma \varDelta \varTheta \varLambda \varXi \varPi \varSigma \varUpsilon \varPhi \varOmega $$

$$ \alpha \beta \gamma \delta \epsilon \zeta \eta \theta \iota \kappa \lambda \mu \nu \xi \omicron \pi \sigma \tau \upsilon \phi \chi \psi \omega  $$
$$ \varepsilon \varkappa \vartheta \thetasym \varpi \varrho \varsigma \varphi \digamma $$

$$ \cancel{5} \bcancel{5} \xcancel{ABC} \sout{abc} \not = $$

$$ \overbrace{a+b+c}^{\text{note}} \underbrace{a+b+c}_{\text{note}} $$

$$ \boxed{\pi=\frac c d} $$

$$ \tag{hi} x+y^{2x} $$

$$ \tag*{hi} x+y^{2x} $$

$$ x^n x_n $$

$$ _u^o $$

$$ a \atop b $$

$$ a\raisebox{0.25em}{b}c $$

$$ \stackrel{!}{=} \overset{!}{=} \underset{!}{=} $$

$$ {=}\mathllap{/\,} \mathrlap{\,/}{=} $$

$$ \left(x^{\smash{2}}\right) (x^2) \sqrt{\smash{y}} $$

$$ \sum_{\mathclap{1 \le i \le i \le n}} x_{ij} $$

$$ \forall \exists \exist \nexists \in \notin \isin \complement \subset $$

$$ \supset \mid \land \lor \ni \therefore \because \mapsto \to \gets  $$

$$ \leftrightarrow \notni \emptyset \empty \varnothing \implies \impliedby $$

$$ \iff \neg \lnot \not $$

$$ \sum \int \iint \iiint \intop \smallint $$

$$ \cdot \cdotp \And \bullet \cap \cup \circ \mp \pm  $$

$$ x \mod a \\ x \pmod a \\ x \pod a $$

$$ \frac{a}{b} {a \over b} \tfrac{a}{b} \dfrac{a}{b} \cfrac{a}{1+\cfrac{1}{b}} $$

$$ \binom{n}{k} \dbinom{n}{k} \tbinom{n}{k}  $$

$$ {n \brace k} {c \choose k} {n\brack k} $$

$$ \sqrt{x} \sqrt[3]{y} $$

$$ \xrightarrow[def]{abc} $$

$$ %comment $$

$$ \infty $$

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

##### 確率密度計算 stats.norm.pdf #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3

x=numpy.arange(start=1,stop=7.1,step=0.1)
y=stats.norm.pdf(x=x,loc=4,scale=0.8)
pyplot.plot(x,y,color='black')
~~~

##### 正規分布に従う乱数生成 stats.norm.rvs #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
# 平均loc 標準偏差scale サンプルサイズsize
sampling_norm = stats.norm.rvs(loc=4,scale=0.8,size=10)
sampling_norm
~~~

#### 第３部 第5章 標本の統計量の性質 ####

##### 10000個標本平均 #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
seaborn.set()

population = stats.norm(loc=4,scale=0.8)
sample_mean_array = numpy.zeros(10000)
numpy.random.seed(1)
for i in range(0,10000):
    sample = population.rvs(size = 10)
    sample_mean_array[i] = scipy.mean(sample)

mean = scipy.mean(sample_mean_array)
std = scipy.std(sample_mean_array)
seaborn.distplot(sample_mean_array,color='black')
~~~

##### サンプルサイズが大きくなると、標本平均は母平均に近づいていく #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
seaborn.set()

population = stats.norm(loc=4,scale=0.8)

size_array = numpy.arange(start=10,stop=100100,step=100)
sample_mean_array_size = numpy.zeros(len(size_array))
numpy.random.seed(1)
for i in range(0,len(size_array)):
    sample = population.rvs(size = size_array[i])
    sample_mean_array_size[i] = scipy.mean(sample)

pyplot.plot(size_array,sample_mean_array_size,color='black')
pyplot.xlabel("sample size")
pyplot.ylabel("sample mean")
~~~

##### 標本平均の標準偏差は、母標準偏差よりも小さい #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
seaborn.set()

# 平均4、標準偏差0.8(分散0.64)の正規分布
population = stats.norm(loc=4,scale=0.8)
size_array = numpy.arange(start=2,stop=102,step=2)
sample_mean_std_array = numpy.zeros(len(size_array))
numpy.random.seed(1)
for i in range(0,len(size_array)):
    sample_mean_array = numpy.zeros(100)
    for m in range(0,100):
        sample = population.rvs(size = size_array[i])
        sample_mean_array[m] = scipy.mean(sample)
    sample_mean_std_array[i] = scipy.std(sample_mean_array,ddof=1)
pyplot.plot(size_array,sample_mean_std_array,color='black')
pyplot.xlabel("sample size")
pyplot.ylabel("mean_std value")
~~~

##### サンプルサイズが大きくなると、標準誤差は小さくなる #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
seaborn.set()

population = stats.norm(loc=4,scale=0.8)

size_array = numpy.arange(start=2,stop=102,step=2)
sample_mean_std_array = numpy.zeros(len(size_array))
numpy.random.seed(1)
for i in range(0,len(size_array)):
    sample_mean_array = numpy.zeros(100)
    for m in range(0,100):
        sample = population.rvs(size = size_array[i])
        sample_mean_array[m] = scipy.mean(sample)
    sample_mean_std_array[i] = scipy.std(sample_mean_array,ddof=1)

standard_error = 0.8 / numpy.sqrt(size_array)

pyplot.plot(size_array,sample_mean_std_array,color='black')
pyplot.plot(size_array,standard_error,color='black',linestyle='dotted')
pyplot.xlabel("sample size")
pyplot.ylabel("mean_std value")
~~~

##### サンプルサイズ大なら、不偏分散は母分散に近い #####

~~~python
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
%matplotlib inline
import seaborn
%precision 3
seaborn.set()

population = stats.norm(loc=4,scale=0.8)

size_array = numpy.arange(start=10,stop=100100,step=100)
unbias_var_array_size = numpy.zeros(len(size_array))
numpy.random.seed(1)
for i in range(0,len(size_array)):
    sample = population.rvs(size = size_array[i])
    unbias_var_array_size[i] = scipy.var(sample,ddof=1)

pyplot.plot(size_array,unbias_var_array_size,color='black')
pyplot.xlabel("sample size")
pyplot.ylabel("unbias var")
~~~

#### 第３部 第6章 正規分布とその応用 ####

