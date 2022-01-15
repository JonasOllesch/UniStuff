import matplotlib.pyplot as plt
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
#---------------------------------------------------------------------------------
