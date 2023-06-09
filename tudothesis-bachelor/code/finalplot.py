import numpy as np
import matplotlib.pyplot as plt


## Functions for m1:
# m_1 in dependence of g_ee and g_1 (with phase 0)

def g_ee_m1_pha0(g_ee, g1, delm_sunsqua, sinsqua_sun):
    denom_1 = (g_ee - g1 * (1 - sinsqua_sun)) / (sinsqua_sun * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)

# m_1 in dependence of g_emu and g_1 (with phase 0)

def g_emu_m1_pha0(g_emu, g1, delm_sunsqua, theta_sun):
    denom_1 = (2*g_emu / (g1 * np.sin(2 * theta_sun)) + 1)
    return np.sqrt(delm_sunsqua/(denom_1**2 - 1))


# m_1 in dependence of g_mumu and g_1 (with phase 0)

def g_mumu_m1_pha0(g_mumu, g1, delm_sunsqua, theta_sun):
    denom_1 = (g_mumu - g1 * np.sin(theta_sun)**2) / (np.cos(theta_sun)**2 * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)


# m_1 in dependence of g_tautau and g_1 (with phase 0)

def g_tautau_m1_pha0(g_tautau, g1, delm_sunsqua, delm_atmsqua):
    denom_1 = g_tautau/g1 #* np.exp(2j * delta)
    return ((delm_sunsqua + delm_atmsqua) / (denom_1**2 - 1))**(1/2)


## Functions to calculate the g1-cutoffs from limits on m1 and/or intersections
#  g1-cutoff for g_ee

def g1_g_ee_cutoff_pha0(g_ee, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.cos(theta_sun)**2 + np.sqrt(1 + delm_sunsqua/m1_lim**2) * np.sin(theta_sun)**2
    return g_ee/denom_1

def g1_g_ee_cutoff_pha90(g_ee, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.cos(theta_sun)**2 - np.sqrt(1 + delm_sunsqua/m1_lim**2) * np.sin(theta_sun)**2
    return g_ee/denom_1


# g1-cutoff for g_emu intersection

def g1_g_emu_cutoff(g_low, g_upper, theta_sun):
    denom_1 = (g_low - g_upper) * np.sin(2 * theta_sun)
    return (g_upper**2 - g_low**2) / denom_1


# g1-cutoff for g_mumu from m1 limit

def g1_g_mumu_cutoff_pha0(g_mumu, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.sin(theta_sun)**2 + np.sqrt(1 + delm_sunsqua / m1_lim**2) * np.cos(theta_sun)**2
    return g_mumu/denom_1

def g1_g_mumu_cutoff_pha90(g_mumu, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.sin(theta_sun)**2 - np.sqrt(1 + delm_sunsqua / m1_lim**2) * np.cos(theta_sun)**2
    return g_mumu/denom_1


# g1-cutoff for g_tautau from m1 limit

def g1_g_tautau_cutoff(g_tautau, m1_lim, delm_sunsqua, delm_atmsqua):
    denom_1 = 1 + (delm_sunsqua + delm_atmsqua) / m1_lim**2
    return g_tautau / np.sqrt(denom_1)


# g_ee g_emu intersection

def g_ee_g_emu_intersect(g_ee, g_emu, theta_sun):
    enum_1  = g_ee / np.sin(theta_sun)**2 - 2 * g_emu / np.sin(2*theta_sun)
    denom_1 = 1 + np.cos(theta_sun)**2 / np.sin(theta_sun)**2
    return enum_1 / denom_1

def g_ee_g_emu_intersect_comp_pos(g_ee, g_emu, theta_sun):
    p = 1/(1 - 2*np.cos(theta_sun)**2) * (2*g_emu / np.sin(2*theta_sun) * np.sin(theta_sun)**4 + g_ee * np.cos(theta_sun)**2)
    q = 1/(1 - 2*np.cos(theta_sun)**2) * (4*g_emu**2 / np.sin(2*theta_sun)**2 * np.sin(theta_sun)**4 - g_ee**2)
    return - p + np.sqrt(p**2 -q)

def g_ee_g_emu_intersect_comp_neg(g_ee, g_emu, theta_sun):
    p = 1/(1 - 2*np.cos(theta_sun)**2) * (2*g_emu / np.sin(2*theta_sun) * np.sin(theta_sun)**4 + g_ee * np.cos(theta_sun)**2)
    q = 1/(1 - 2*np.cos(theta_sun)**2) * (4*g_emu**2 / np.sin(2*theta_sun)**2 * np.sin(theta_sun)**4 - g_ee**2)
    return - p - np.sqrt(p**2 -q)


# g1-cutoff for g_ee intersection

def g1_g_ee_cutoff(g_low, g_upper, theta_sun):
    denom_1 = 2 * (g_upper - g_low) * np.cos(theta_sun)**2
    return (g_upper**2 - g_low**2) / denom_1



delm_sunsqua = 7.53 * 10**(-5)                   # new value from the PDG                                        
delm_atmsqua = 2.453 * 10**(-5)                  # new value from the PDG

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.546))/2
sinsqua_sun = np.sin(theta_sun)**2   

g1 = np.logspace(start = -12, stop = -3, num = 2 * 10**6, base = 10.0)      

# upper limit on m1

m1_lim = 0.8

# lower bounds:
g_ee_low     = 3 * 10**(-7)
g_emu_low    = 3 * 10**(-7)
g_mumu_low   = 3 * 10**(-7)
g_tautau_low = 3 * 10**(-7)

g_ee_neglow     = -3 * 10**(-7)
g_emu_neglow    = -3 * 10**(-7)
g_mumu_neglow   = -3 * 10**(-7)
g_tautau_neglow = -3 * 10**(-7)
# upper bounds:
g_ee_upper     = 2 * 10**(-5)               # updated upper bounds on g_ee
g_emu_upper    = 2 * 10**(-5)
g_mumu_upper   = 2 * 10**(-5)
g_tautau_upper = 2 * 10**(-5)

g_ee_negupper     = -2 * 10**(-5)
g_emu_negupper    = -2 * 10**(-5)
g_mumu_negupper   = -2 * 10**(-5)
g_tautau_negupper = -2 * 10**(-5)


# print(g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun))

print('g_ee cutoff with phase 0: ',g1_g_ee_cutoff_pha0(g_ee_low, m1_lim, delm_sunsqua, theta_sun), g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun))
print('g_ee cutoff with phase 90: ',g1_g_ee_cutoff_pha90(g_ee_low, m1_lim, delm_sunsqua, theta_sun), g1_g_ee_cutoff_pha90(g_ee_upper, m1_lim, delm_sunsqua, theta_sun))



# limited logspaces from cutoffs

### for g1_ee
num = 2 * 10**6  # Number of elements in the logspace array

# lower phase 0 cutoff
g1_ee_low_pha0 = np.logspace(-12, np.log10(g1_g_ee_cutoff_pha0(g_ee_low, m1_lim, delm_sunsqua, theta_sun)), num)

# lower phase 90 cutoff
g1_ee_low_pha90 = np.logspace(np.log10(g1_g_ee_cutoff_pha90(g_ee_low, m1_lim, delm_sunsqua, theta_sun)), -3, num)


# upper phase 0 cutoff
g1_ee_upper_pha0 = np.logspace(-12, np.log10(g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), num)

# upper phase 90 cutoff
g1_ee_upper_pha90 = np.logspace(np.log10(g1_g_ee_cutoff_pha90(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), -3, num)


### for g_emu
print('g_emu cutoff: ', g1_g_emu_cutoff(g_emu_low, g_emu_negupper, theta_sun))


g1_emu_cutoff = np.logspace(-12, np.log10(g1_g_emu_cutoff(g_emu_low, g_emu_negupper, theta_sun)), num)


### for g_mumu

# lower phase 90 cutoff
g1_mumu_low_pha90 = np.logspace(-12, np.log10(g1_g_mumu_cutoff_pha90(g_mumu_neglow, m1_lim, delm_sunsqua, theta_sun)), num)
print(g1_mumu_low_pha90)

# upper phase 0 cutoff
g1_mumu_upper_pha0 = np.logspace(-12, np.log10(g1_g_mumu_cutoff_pha0(g_mumu_upper, m1_lim, delm_sunsqua, theta_sun)), num)


# for g_tautau

#lower phase 0 cutoff
g1_tautau_low = np.logspace(-12, np.log10(g1_g_tautau_cutoff(g_tautau_low, m1_lim, delm_sunsqua, delm_atmsqua)), num)
print(g1_tautau_low)

# upper phase 0 cutoff
g1_tautau_upper = np.logspace(-12, np.log10(g1_g_tautau_cutoff(g_tautau_upper, m1_lim, delm_sunsqua, delm_atmsqua)), num)



### Intersections between g_ee and g_emu

# Intersection between lower bound on g_eumu and positive upper bound on g_ee

g1_intsect_lower_upper = g_ee_g_emu_intersect(g_ee_upper, g_emu_low, theta_sun)
print('Intersection between lower bound on g_eumu and positive upper bound on g_ee: ', g1_intsect_lower_upper)

# Intersection between (negative) upper bound on g_emu and positive lower bound on g_ee

g1_intsect_upper_lower = g_ee_g_emu_intersect(g_ee_low, g_emu_negupper, theta_sun)

print('Intersection between (negative) upper bound on g_emu and positive lower bound on g_ee: ', g1_intsect_upper_lower)

# Intersection between (negative) upper bound on g_emu and negative lower bound on g_ee

g1_intsect_upper_neglower = g_ee_g_emu_intersect(g_ee_neglow, g_emu_negupper, theta_sun)

print('Intersection between (negative) upper bound on g_emu and negative lower bound on g_ee: ', g1_intsect_upper_neglower)

# Intersection between (negative) upper bound on g_emu and positive upper bound on g_ee:

g1_intsect_upper_upper = g_ee_g_emu_intersect_comp_neg(g_ee_upper, g_emu_negupper, theta_sun)
print('Intersection between (negative) upper bound on g_emu and positive upper bound on g_ee: ', g1_intsect_upper_upper)


### Intersections between different bounds on g_ee

g_ee_cutoff_neglower = g1_g_ee_cutoff(g_ee_neglow, g_ee_upper, theta_sun)
print('Intersection between negative lower bound and upper bound on g_ee: ', g_ee_cutoff_neglower)

g_ee_cutoff_lower = g1_g_ee_cutoff(g_ee_low, g_ee_upper, theta_sun)
print('Intersection between positive lower bound and upper bound on g_ee: ', g_ee_cutoff_lower)


## Plots
# g_ee lower bound with phase 0

#plt.plot(g1_ee_low_pha0, g_ee_m1_pha0(g_ee_low, g1_ee_low_pha0, delm_sunsqua, sinsqua_sun), color='blue', label=r'positive bounds on $g_{ee}$')
#plt.plot(g1_ee_low_pha90, g_ee_m1_pha0(g_ee_low, g1_ee_low_pha90, delm_sunsqua, sinsqua_sun), color='blue', label=r'positive bounds on $g_{ee}$')
#plt.plot(g1, g_ee_m1_pha0(g_ee_neglow, g1, delm_sunsqua, sinsqua_sun), color='teal',label=r'negative bounds on $g_{ee}$')

# g_ee lower bound with phase pi/2

#plt.plot(g1, g_ee_m1_pha90(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='yellow',label=r'lower bound on $g_{ee}$')  # It seems like I don't need this

# g_ee upper bound

#plt.plot(g1_ee_upper_pha0, g_ee_m1_pha0(g_ee_upper, g1_ee_upper_pha0, delm_sunsqua, sinsqua_sun), color='blue')
#plt.plot(g1_ee_upper_pha90, g_ee_m1_pha0(g_ee_upper, g1_ee_upper_pha90, delm_sunsqua, sinsqua_sun), color='blue')
#plt.plot(g1, g_ee_m1_pha0(g_ee_negupper, g1, delm_sunsqua, sinsqua_sun), color='teal')



# g_eμ lower bound

#plt.plot(g1_emu_cutoff, g_emu_m1_pha0(g_emu_low, g1_emu_cutoff, delm_sunsqua, theta_sun), color='orange', linestyle='dashed', label=r'lower bound on $g_{e \mu^\prime}$') ### Der ist gut!!
#plt.plot(g1, g_emu_m1_pha0(g_emu_neglow, g1, delm_sunsqua, theta_sun), color='yellow', linestyle='dashed')

# g_eμ upper bound

#plt.plot(g1, g_emu_m1_pha0(g_emu_upper, g1, delm_sunsqua, theta_sun), color='orange', linestyle='dashed')
#plt.plot(g1_emu_cutoff, g_emu_m1_pha0(g_emu_negupper, g1_emu_cutoff, delm_sunsqua, theta_sun), color='yellow', linestyle='dashed', label=r'upper bound on $g_{e \mu^\prime}$') ### Der auch, der macht den tollen Knick!


# g_μμ lower bound

#plt.plot(g1, g_mumu_m1_pha0(g_mumu_low, g1, delm_sunsqua, theta_sun), color='purple', linestyle='dotted', label=r'bounds on $g_{\mu^\prime \mu^\prime}$')
#plt.plot(g1_mumu_low_pha90, g_mumu_m1_pha0(g_mumu_neglow, g1_mumu_low_pha90, delm_sunsqua, theta_sun), color='pink', linestyle='dotted', label=r'lower bound on $g_{\mu^\prime \mu^\prime}$')

# g_μμ upper bound

#plt.plot(g1_mumu_upper_pha0, g_mumu_m1_pha0(g_mumu_upper, g1_mumu_upper_pha0, delm_sunsqua, theta_sun), color='purple', linestyle='dotted', label=r'upper bound on $g_{\mu^\prime \mu^\prime}$')
#plt.plot(g1, g_mumu_m1_pha0(g_mumu_negupper, g1, delm_sunsqua, theta_sun), color='pink', linestyle='dotted')

### final plot

## setting the logspaces

ee_intersect_neglow = np.logspace(-12, np.log10(g_ee_cutoff_neglower), num)
ee_neglow_part      = np.logspace(np.log10(g1_intsect_upper_neglower), np.log10(g_ee_cutoff_neglower), num)
emu_negupper_part   = np.logspace(np.log10(g1_intsect_upper_neglower), np.log10(g1_intsect_upper_lower), num)
ee_low_part         = np.logspace(np.log10(g1_intsect_upper_lower), np.log10(g_ee_cutoff_lower), num)
ee_upper_part       = np.logspace(np.log10(g_ee_cutoff_lower), np.log10(g1_intsect_upper_upper), num)
emu_negupper_part_2 = np.logspace(np.log10(g1_intsect_upper_upper), np.log10(g1_g_emu_cutoff(g_emu_low, g_emu_negupper, theta_sun)), num)
emu_low_part        = np.logspace(np.log10(g1_intsect_lower_upper), np.log10(g1_g_emu_cutoff(g_emu_low, g_emu_negupper, theta_sun)), num)
ee_upper_part_final = np.logspace(np.log10(g1_intsect_lower_upper), np.log10(g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), num)
print(ee_intersect_neglow)


# g_ττ lower bound
m1_lim_constant = np.full_like(g1, m1_lim)

plt.plot(g1_tautau_low, g_tautau_m1_pha0(g_tautau_low, g1_tautau_low, delm_sunsqua, delm_atmsqua), color='purple', linestyle='dashdot', label=r'bounds on $g_{\tau^\prime \tau^\prime}$')

plt.plot(ee_intersect_neglow, g_ee_m1_pha0(g_ee_upper, ee_intersect_neglow, delm_sunsqua, sinsqua_sun), color='purple')
plt.plot(ee_neglow_part, g_ee_m1_pha0(g_ee_neglow, ee_neglow_part, delm_sunsqua, sinsqua_sun), color='purple')

plt.plot(emu_negupper_part, g_emu_m1_pha0(g_emu_negupper, emu_negupper_part, delm_sunsqua, theta_sun), color='purple')
plt.plot(ee_low_part, g_ee_m1_pha0(g_ee_low, ee_low_part, delm_sunsqua, sinsqua_sun), color='purple')
plt.plot(ee_upper_part, g_ee_m1_pha0(g_ee_upper, ee_upper_part, delm_sunsqua, sinsqua_sun), color='purple')
plt.plot(emu_negupper_part_2, g_emu_m1_pha0(g_emu_negupper, emu_negupper_part_2, delm_sunsqua, theta_sun), color='purple')
plt.plot(emu_low_part, g_emu_m1_pha0(g_emu_low, emu_low_part, delm_sunsqua, theta_sun), color='purple')
plt.plot(ee_upper_part_final, g_ee_m1_pha0(g_ee_upper, ee_upper_part_final, delm_sunsqua, sinsqua_sun), color='purple')



# fill-ins
num1 = 1000
g_tautau_m1_cutoff = np.logspace(-12, np.log10(g1_g_tautau_cutoff(g_tautau_low, m1_lim, delm_sunsqua, delm_atmsqua)), num1)
m1_g_ee_fillin     = np.logspace(np.log10(g1_g_tautau_cutoff(g_tautau_low, m1_lim, delm_sunsqua, delm_atmsqua) + 10**(-9)), np.log10(g_ee_cutoff_neglower), num1)


plt.fill_between(g_tautau_m1_cutoff, g_tautau_m1_pha0(g_tautau_low, g_tautau_m1_cutoff, delm_sunsqua, delm_atmsqua), g_ee_m1_pha0(g_ee_upper, g_tautau_m1_cutoff, delm_sunsqua, sinsqua_sun), color='purple', alpha=0.5)

plt.fill_between(m1_g_ee_fillin, np.full_like(m1_g_ee_fillin, m1_lim), g_ee_m1_pha0(g_ee_upper, m1_g_ee_fillin, delm_sunsqua, sinsqua_sun), color='purple', alpha=0.5)
#plt.fill_between(ee_neglow_part, np.full_like(ee_neglow_part, m1_lim), g_ee_m1_pha0(g_ee_neglow, ee_neglow_part, delm_sunsqua, sinsqua_sun), color='purple', alpha=0.5)


plt.plot(g1, m1_lim_constant, color='blue')


plt.plot

plt.xlabel(r'$g_1$')
plt.xlim(10**(-11), 10**(-3))
plt.xscale('log')

plt.ylabel(r'$m_1$')
plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/finalplot.pdf')
plt.clf()