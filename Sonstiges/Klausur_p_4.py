import matplotlib.pyplot as plt
import numpy as np

def horisontallineat(x):
    return x

Messung = np.array(np.genfromtxt('Klausur_p_4_Daten.txt'))

x = np.linspace(0,35,20)
y = np.zeros((20))
y[:] = horisontallineat(22)

durchschnitt = np.mean(Messung[:,1])
durchschnitt_ = np.zeros((20))
durchschnitt_[:] = horisontallineat(durchschnitt)
standartabweichung = np.std(Messung[:,1])
dpstd = np.zeros((20))
dpstd[:] = horisontallineat(durchschnitt+standartabweichung)
dmstd = np.zeros((20))
dmstd[:] = horisontallineat(durchschnitt-standartabweichung)



plt.plot(x,dpstd,color='lightgreen',label="std")
plt.plot(x,dmstd,color='lightgreen')
plt.plot(x,y,color='red',label="Bestehensgrenze")
plt.plot(x,durchschnitt_,color='green',label="Durchschnitt")
plt.scatter(Messung[:,0],Messung[:,1],c = 'blue',s = 6,label = "Messdaten")
plt.xlabel("Punkte")
plt.legend()
plt.tight_layout()

plt.grid()
plt.savefig('Klausur_p_4.pdf')
plt.clf()