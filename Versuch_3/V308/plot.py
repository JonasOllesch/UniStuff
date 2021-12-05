import matplotlib.pyplot as pyplot
import numpy as np

WerteHysterese = np.array(np.genfromtxt('WerteHysterese.txt'))
WerteHysterese[:,1] = WerteHysterese[:,1] / 1000 #von mT in T umrechnen

x = WerteHysterese[:,0]
y = WerteHysterese[:,1]

pyplot.scatter(x, y, color='blue',s=15, label=r'$B$')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$I \mathbin{/} \unit{\ampere}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.show()
pyplot.savefig('build/Hysteresplot.pdf')
pyplot.clf()

#-----------------------------------
WertelangeSpule = np.array(np.genfromtxt('WertelangeSpule.txt'))
WertelangeSpule[:,1] = WertelangeSpule[:,1] / 1000 #von mT in T umrechnen
WertelangeSpule[:,0] = WertelangeSpule[:,0] / 100 # von cm in m

x = WertelangeSpule[:,0]
y = WertelangeSpule[:,1]

pyplot.scatter(x, y, color='blue',s=15, label=r'$B$')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$M \mathbin{/} \unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.show()
pyplot.savefig('build/WertelangeSpule.pdf')
pyplot.clf()

#-----------------------------------
WerteSpulenPaar10 = np.array(np.genfromtxt('WerteSpulenPaar10.txt'))
WerteSpulenPaar10[:,1] = WerteSpulenPaar10[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar10[:,0] = WerteSpulenPaar10[:,0] / 100 # von cm in m

x = WerteSpulenPaar10[:,0]
y = WerteSpulenPaar10[:,1]

pyplot.scatter(x, y, color='blue',s=15, label=r'$B$')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$M \mathbin{/} \unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar10.pdf')
pyplot.clf()
#-----------------------------------
WerteSpulenPaar20 = np.array(np.genfromtxt('WerteSpulenPaar20.txt'))
WerteSpulenPaar20[:,1] = WerteSpulenPaar20[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar20[:,0] = WerteSpulenPaar20[:,0] / 100 # von cm in m

x = WerteSpulenPaar20[:,0]
y = WerteSpulenPaar20[:,1]

pyplot.scatter(x, y, color='blue',s=15, label=r'$B$')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$M \mathbin{/} \unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar20.pdf')
pyplot.clf()
#-----------------------------------
WerteSpulenPaar15 = np.array(np.genfromtxt('WerteSpulenPaar20.txt'))
WerteSpulenPaar15[:,1] = WerteSpulenPaar15[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar15[:,0] = WerteSpulenPaar15[:,0] / 100 # von cm in m

x = WerteSpulenPaar15[:,0]
y = WerteSpulenPaar15[:,1]

pyplot.scatter(x, y, color='blue',s=15, label=r'$B$')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$M \mathbin{/} \unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.show()
pyplot.savefig('build/WerteSpulenPaar15.pdf')
pyplot.clf()

