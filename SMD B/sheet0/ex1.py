from struct import unpack
import numpy as np
import matplotlib.pyplot as plt

height, weight = np.genfromtxt("height_weight.txt", unpack=True)

fig, axs = plt.subplots(3,2)

axs[0,0].hist(weight, bins = 5, lw = 2)
axs[0,0].set_title('bins = 5')

axs[1,0].hist(weight, bins = 10, lw = 2)
axs[1,0].set_title('bins = 10')

axs[2,0].hist(weight, bins = 15, lw = 2)
axs[2,0].set_title('bins = 15')

axs[0,1].hist(weight, bins = 20, lw = 2)
axs[0,1].set_title('bins = 20')

axs[1,1].hist(weight, bins = 30, lw = 2)
axs[1,1].set_title('bins = 30')

axs[2,1].hist(weight, bins = 50, lw = 2)
axs[2,1].set_title('bins = 50')

plt.tight_layout()

plt.savefig('histogram_weight.pdf')
plt.close()


fig, axs = plt.subplots(3,2)

axs[0,0].hist(height, bins = 5, lw = 2)
axs[0,0].set_title('bins = 5')

axs[1,0].hist(height, bins = 10, lw = 2)
axs[1,0].set_title('bins = 10')

axs[2,0].hist(height, bins = 15, lw = 2)
axs[2,0].set_title('bins = 15')

axs[0,1].hist(height, bins = 20, lw = 2)
axs[0,1].set_title('bins = 20')

axs[1,1].hist(height, bins = 30, lw = 2)
axs[1,1].set_title('bins = 30')

axs[2,1].hist(height, bins = 50)
axs[2,1].set_title('bins = 50')

plt.savefig('histogram_height.pdf')
plt.close()

int = np.random.randint(1, 100, 10**5)

fig,axs= plt.subplots(3,2)


axs[0,0].hist(np.log(int), bins = 20, histtype = 'step')
axs[0,0].set_title('bins = 20')

axs[1,0].hist(np.log(int), bins = 50, histtype = 'step')
axs[1,0].set_title('bins = 50')

axs[2,0].hist(np.log(int), bins = 5, histtype = 'step')
axs[2,0].set_title('bins = 5')

axs[1,1].hist(np.log(int), bins = 100, histtype = 'step')
axs[1,1].set_title('bins = 100')

plt.tight_layout()
plt.savefig('randomint.pdf')
plt.close()

x = np.linspace(-5,15,10000)

y = (x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10)

plt.plot(x,y, c = 'b')
plt.xlim(0,11)
plt.xticks(np.arange(0,11,step=1))
plt.ylim(-10000,15000)
plt.savefig('HoeMa_I_Blatt_2.pdf')

