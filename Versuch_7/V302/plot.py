import matplotlib.pyplot as pyplot
import numpy as np

output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(str(i))
            my_file.write('\n')
    except:
        my_file.write(str(Wert))
        my_file.write('\n')

    return 0

def r_x_berchnen(R2,R3,R4):
    return R2*(R3/R4)

def l_x_berechnen(L2,R3,R4):
    return L2*(R3/R4)

def theorie_kurve(o):
    tmp = (o**2-1)**2
    tmp2 = (1-o**2)**2
    tmp3 = 9*(o**2)   
    return  np.sqrt((1/9)*(tmp/(tmp2+tmp3)))
#-------------------------------------------------------------------------------

#Wheatstonemessbrücke
#zu Wert 12
R1_1 = r_x_berchnen(1000, 282, 718)
R1_2 = r_x_berchnen(664, 371, 629)


#zu Wert 14
R1_3 = r_x_berchnen(1000, 475, 525)
R1_4 = r_x_berchnen(664, 576, 424)


#Induktivitätsmessbrücke
#zu Wert 16
R3_1 = r_x_berchnen(49, 903, 97)
L3_1 = l_x_berechnen(0.146, 903, 97)



R3_2 = r_x_berchnen(63, 871, 129)   
L3_2 = l_x_berechnen(0.201, 871, 129)

#Maxwellbrücke
#zu Wert 16
R_2_max = 332
R_3_max = 662
R_4_max = 448
C_4_max = 597*10**(-9)

R_x_max = (R_2_max*R_3_max)/R_4_max
L_x_max = R_2_max*R_3_max*C_4_max

#Kapazitätsbrücke
#zu Wert 9
R2_1 = r_x_berchnen(281, 632, 368)
C2_1 = (750*10**-9)*(368/632)

R2_2 = r_x_berchnen(347, 582, 418)
C2_2 = (597*10**-9)*(418/582)

x1=np.linspace(0.1,2,1000)
x = np.linspace(2,100,1000)
y1 = theorie_kurve(x1)
y= theorie_kurve(x)
pyplot.plot(x1, y1,color='blue')
pyplot.plot(x, y,color='blue',label='Theorie')
pyplot.xlim(0.1,100)
pyplot.ylim(0,0.35)
pyplot.xscale('log')
pyplot.xlabel(r'$\log{Ω}$')
pyplot.ylabel(r'$\frac{U_{Br}}{U_S}$')


pyplot.legend()
pyplot.grid()
pyplot.tight_layout()
pyplot.savefig('build/Theorie.pdf')
pyplot.clf()

#---------------------------------------------------------------------------------
