# Draw samples from the distribution:
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 1.0, 0.5 # mean and standard deviation
offset = 0.0
s = np.random.lognormal(mu, sigma, 1000) + offset

# Display the histogram of the samples, along with
# the probability density function:


plt.figure(1)
 
count, bins, ignored = plt.hist(s, 50,normed=True, align='mid')

x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))

plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')

plt.show()

