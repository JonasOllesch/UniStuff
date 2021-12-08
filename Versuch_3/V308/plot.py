import matplotlib.pyplot as pyplot
import numpy as np

WerteHysterese = np.array(np.genfromtxt('WerteHysterese.txt'))#Dass das alles blau ist, ist irgendwie doof, lass da morgen mal ne bessere Möglichkeit suchen :D
WerteHysterese[:,1] = WerteHysterese[:,1] / 1000 #von mT in T umrechnen

x = WerteHysterese[:,0]
y = WerteHysterese[:,1]

x_ =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
y_ =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(11,31):
    y_[i-11] =y[i]
    x_[i-11] = x[i] 
pyplot.scatter(x_, y_, color='blue',s=10, label="Hysterese 2")

x_ =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y_ =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
for i in range(31,51):
    y_[i-31] =y[i]
    x_[i-31] = x[i] 
pyplot.scatter(x_, y_, color='red',s=10, label="Hysterese 3")


x_ = [0,1,2,3,4,5,6,7,8,9,10]
y_ = [0,1,2,3,4,5,6,7,8,9,10]
for i in range(0,11):
    y_[i] =y[i]
    x_[i] = x[i] 
pyplot.scatter(x_, y_, color='#4dff00',s=10, label="Neukurve")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$I \mathbin{/}\unit{\ampere}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
#pyplot.show()
pyplot.xlim(-10,10)
pyplot.xticks(np.arange(-10,10.1,1))
pyplot.ylim(-1,1)
pyplot.yticks(np.arange(-1,1,0.25))
pyplot.savefig('build/Hystereseplot.pdf')
pyplot.clf()
#print("Hysterese Ende")
#-----------------------------------


WertelangeSpule = np.array(np.genfromtxt('WertelangeSpule.txt'))
WertelangeSpule[:,1] = WertelangeSpule[:,1] / 1000 #von mT in T umrechnen
WertelangeSpule[:,0] = WertelangeSpule[:,0] / 100 # von cm in m

x = WertelangeSpule[:,0]
y = -WertelangeSpule[:,1]

pyplot.scatter(x, y, color='blue',s=15, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.071,0.01))
pyplot.yticks(np.arange(0,0.0031,0.0005))
pyplot.xlim(0,0.07)
pyplot.ylim(0,0.003)
#pyplot.show()
#x = np.linspace(0.00001,0.7,1000)
#pyplot.plot(x,((300*4*np.pi*(10**(-7)))*(1/0.18))*x/x,color='red',label="Theorie")
#pyplot.legend("Theorie")
#print(((300*4*np.pi*(10**(-7)))*(1/0.18)) , " der exakte Wert")

pyplot.savefig('build/langeSpule.pdf', bbox_inches='tight')
pyplot.clf()

#-----------------------------------
x = np.linspace(0,0.13,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+x**2)**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.10)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")

WerteSpulenPaar10 = np.array(np.genfromtxt('WerteSpulenPaar10.txt'))
WerteSpulenPaar10[:,1] = WerteSpulenPaar10[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar10[:,0] = WerteSpulenPaar10[:,0] / 100 # von cm in m

x = WerteSpulenPaar10[:,0]
y = WerteSpulenPaar10[:,1]

pyplot.scatter(x, y, color='blue',s=15, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.131,0.01))
pyplot.yticks(np.arange(0,0.0051,0.0005))
pyplot.xlim(0,0.13)
pyplot.ylim(0,0.005)
#pyplot.show()
pyplot.savefig('build/SpulenPaar10.pdf')
pyplot.clf()
#-----------------------------------
x = np.linspace(0,0.13,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x)**2)**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.10)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")

WerteSpulenPaar10 = np.array(np.genfromtxt('WerteSpulenPaar10.txt'))
WerteSpulenPaar10[:,1] = WerteSpulenPaar10[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar10[:,0] = WerteSpulenPaar10[:,0] / 100 # von cm in m

x = WerteSpulenPaar10[:,0]+0.015#825
y = WerteSpulenPaar10[:,1]

pyplot.scatter(x, y, color='blue',s=15, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.131,0.01))
pyplot.yticks(np.arange(0,0.0051,0.0005))
pyplot.xlim(0,0.13)
pyplot.ylim(0,0.0055)

#pyplot.show()
pyplot.savefig('build/SpulenPaar10korrektur.pdf')
pyplot.clf()

#-----------------------------------
x = np.linspace(0,0.23,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+x**2)**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.15)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")


WerteSpulenPaar15 = np.array(np.genfromtxt('WerteSpulenPaar15.txt'))
WerteSpulenPaar15[:,1] = WerteSpulenPaar15[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar15[:,0] = WerteSpulenPaar15[:,0] / 100 # von cm in m
#print(WerteSpulenPaar15)
x = WerteSpulenPaar15[:,0]
y = WerteSpulenPaar15[:,1]

pyplot.scatter(x, y, color='blue',s=10, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.2316,0.05))
pyplot.yticks(np.arange(0,0.0051,0.0005))
pyplot.xlim(0,0.235)
pyplot.ylim(0,0.005)

#pyplot.show()
pyplot.savefig('build/SpulenPaar15.pdf')
pyplot.clf()

#-----------------------------------
x = np.linspace(0,0.23,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x**2))**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.15)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")


WerteSpulenPaar15 = np.array(np.genfromtxt('WerteSpulenPaar15.txt'))
WerteSpulenPaar15[:,1] = WerteSpulenPaar15[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar15[:,0] = WerteSpulenPaar15[:,0] / 100 # von cm in m
#print(WerteSpulenPaar15)
x = WerteSpulenPaar15[:,0]+0.015#5
y = WerteSpulenPaar15[:,1]

pyplot.scatter(x, y, color='blue',s=10, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.2316,0.05))
pyplot.yticks(np.arange(0,0.0051,0.0005))
pyplot.xlim(0,0.235)
pyplot.ylim(0,0.005)
#pyplot.show()
pyplot.savefig('build/SpulenPaar15korrektur.pdf')
pyplot.clf()

#-----------------------------------
x = np.linspace(0,0.26,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x**2))**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.20)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")


WerteSpulenPaar20 = np.array(np.genfromtxt('WerteSpulenPaar20.txt'))
WerteSpulenPaar20[:,1] = WerteSpulenPaar20[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar20[:,0] = WerteSpulenPaar20[:,0] / 100 # von cm in m

x = WerteSpulenPaar20[:,0]+0.015#2173
y = WerteSpulenPaar20[:,1]

pyplot.scatter(x, y, color='blue',s=15, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.251,0.05))
pyplot.yticks(np.arange(0,0.0046,0.0005))
pyplot.xlim(0,0.235)
#pyplot.show()
pyplot.savefig('build/SpulenPaar20korrektur.pdf')
pyplot.clf()


#-----------------------------------
x = np.linspace(0,0.26,1000)
pyplot.plot(x,((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+x**2)**(3/2))+((100*4*np.pi*(10**(-7)))*(4/2))*((0.0625)**2)/((0.0625**2+(x-0.20)**2)**(3/2)),color='red',label="Theorie")
pyplot.legend("Theorie")


WerteSpulenPaar20 = np.array(np.genfromtxt('WerteSpulenPaar20.txt'))
WerteSpulenPaar20[:,1] = WerteSpulenPaar20[:,1] / 1000 #von mT in T umrechnen
WerteSpulenPaar20[:,0] = WerteSpulenPaar20[:,0] / 100 # von cm in m

x = WerteSpulenPaar20[:,0]
y = WerteSpulenPaar20[:,1]

pyplot.scatter(x, y, color='blue',s=15, label="Messung")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$x \mathbin{/}\unit{\meter}$')
pyplot.ylabel(r'$B \mathbin{/}\unit{\tesla}$')
pyplot.xticks(np.arange(0,0.251,0.05))
pyplot.yticks(np.arange(0,0.0046,0.0005))
pyplot.xlim(0,0.235)

#pyplot.show()
pyplot.savefig('build/SpulenPaar20.pdf')
pyplot.clf()