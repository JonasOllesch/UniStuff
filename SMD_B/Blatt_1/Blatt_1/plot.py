import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)
a_0 = np.random.normal(1.0, 0.2, size=1000)
a_1 = np.random.normal(1.0, 0.2, size=1000)

#print(a_0)



covariance_matrix = [[0.2**2,0.2**2*0.8],[0.2**2*0.8,0.2**2]]
a0_a1_matrix = np.random.multivariate_normal([1.0,1.0], covariance_matrix, size=1000)



plt.scatter(a_0,a_1,s=8, c='blue')
plt.ylabel(r'$a_1$')
plt.ylim(0,1.8)
plt.xlabel(r'$a_0$')
plt.xlim(0,2)
#plt.axis('equal')
plt.grid()
plt.tight_layout()
#plt.legend("ohne Korrelation")

plt.savefig('build/Graph_a.pdf')
plt.clf()


plt.scatter(a0_a1_matrix[:,0],a0_a1_matrix[:,1],s=8, c='blue')
plt.ylabel(r'$a_1$')
plt.ylim(0,1.8)
plt.xlabel(r'$a_0$')
plt.xlim(0,2)
#plt.axis('equal')
plt.grid()
plt.tight_layout()
#plt.legend("mit Korrelation")

plt.savefig('build/Graph_b.pdf')
plt.clf()

y_mean_mit_korrelation = np.zeros(3)
y_std_mit_korrelation = np.zeros(3)
y_mean_ohne_korrelation = np.zeros(3)
y_std_ohne_korrelation = np.zeros(3)


x_werte = [-3,0,3]

for i in range(0,3):
    y_mean_mit_korrelation[i] = np.mean(a0_a1_matrix[:,0]+a0_a1_matrix[:,1]*x_werte[i])
    y_std_mit_korrelation[i] = np.std(a0_a1_matrix[:,0]+a0_a1_matrix[:,1]*x_werte[i])
    y_mean_ohne_korrelation[i] = np.mean(a_0[:]+a_1[:]*x_werte[i])
    y_std_ohne_korrelation[i] = np.std(a_0[:]+a_1[:]*x_werte[i])


print(y_mean_ohne_korrelation)
print(y_std_ohne_korrelation)
print(y_mean_mit_korrelation)
print(y_std_mit_korrelation)
