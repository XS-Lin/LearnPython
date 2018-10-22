##### サンプルサイズが大きくなると、標準誤差は小さくなる #####
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot
import seaborn
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
pyplot.show()