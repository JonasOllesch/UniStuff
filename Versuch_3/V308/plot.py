import matplotlib.pyplot as pyplot
import numpy as np

WerteHysterese = np.array(np.genfromtxt('WerteHysterese.txt'))
WerteHysterese[:,1] = WerteHysterese[:,1] / 1000 #von mT in T umrechnen

x = WerteHysterese[:,0]
y = WerteHysterese[:,1]

pyplot.scatter(x, y, s=15)
pyplot.legend()
#pyplot.plot(x, y, '--', color='red')
pyplot.show()
pyplot.savefig('build/Hystereseplot.pdf')

WerteHysterese = np.array(np.genfromtxt('WerteHysterese.txt'))
WerteHysterese[:,1] = WerteHysterese[:,1] / 1000 #von mT in T umrechnen
pyplot.clf()





WertelangeSpule = np.array(np.genfromtxt('WertelangeSpule.txt'))
WertelangeSpule[:,1] = WertelangeSpule[:,1] / 1000 #von mT in T umrechnen
WertelangeSpule[:,1] = WertelangeSpule[:,1] / 100   # von cm in m umrechnen

x = WertelangeSpule[:,0]
y = WertelangeSpule[:,1]

pyplot.scatter(x, y, s=15)
pyplot.legend()
#pyplot.plot(x, y, '--', color='red')
pyplot.show()
pyplot.savefig('build/langeSpule.pdf')
pyplot.clf()




WerteSpulenPaar10 = np.array(np.genfromtxt('WerteSpulenPaar10.txt'))
WerteSpulenPaar10[:,1] = WerteSpulenPaar10[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar10[:,1] = WerteSpulenPaar10[:,1] / 100   # von cm in m umrechnen

x = WerteSpulenPaar10[:,0]
y = WerteSpulenPaar10[:,1]

pyplot.scatter(x, y, s=15)
pyplot.legend()
#pyplot.plot(x, y, '--', color='red')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar10.pdf')
pyplot.clf()



WerteSpulenPaar20 = np.array(np.genfromtxt('WerteSpulenPaar20.txt'))
WerteSpulenPaar20[:,1] = WerteSpulenPaar20[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar20[:,1] = WerteSpulenPaar20[:,1] / 100   # von cm in m umrechnen

x = WerteSpulenPaar20[:,0]
y = WerteSpulenPaar20[:,1]

pyplot.scatter(x, y, s=15)
pyplot.legend()
#pyplot.plot(x, y, '--', color='red')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar20.pdf')
pyplot.clf()




WerteSpulenPaar15 = np.array(np.genfromtxt('WerteSpulenPaar15.txt'))
WerteSpulenPaar15[:,1] = WerteSpulenPaar15[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar15[:,1] = WerteSpulenPaar15[:,1] / 100   # von cm in m umrechnen

x = WerteSpulenPaar15[:,0]
y = WerteSpulenPaar15[:,1]

pyplot.scatter(x, y, s=15)
pyplot.legend()
#pyplot.plot(x, y, '--', color='red')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar15.pdf')
pyplot.clf()

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
