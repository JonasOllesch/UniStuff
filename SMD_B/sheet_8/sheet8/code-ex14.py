import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)

def create_response_matrix(e,n):  #creating the matrix 
    A = np.eye(n,M=n,k=-1)*e+np.eye(n,M=n,k=0)*(1-2*e)+np.eye(n,M=n,k=1)*e
    A[0][0]= 1-e
    A[n-1][n-1]= 1-e
    return A

def bubblesort(A,B):    #sorting the eigenvalues with the best algorithm ever
    werte = np.copy(A)
    Matrix = np.copy(B)
    for i in range(len(werte)):
        for j in range(len(werte)-1):
            if werte[j] < werte[j+1]:
                tmp = werte[j]
                werte[j]= werte[j+1]
                werte[j+1]=tmp
                del tmp
                tmpm = np.copy(Matrix[:,j])
                Matrix[:,j]= np.copy(Matrix[:,j+1])
                Matrix[:,j+1] = np.copy(tmpm)
                del tmpm
                #print(i,Matrix[0,:])
    return werte, Matrix


#a)
f = np.array([193, 485, 664, 763, 804, 805, 779, 736, 684, 626, 566, 508, 452, 400, 351, 308, 268, 233, 202, 173])#initialize variables
A = create_response_matrix(0.23, len(f))
print(np.linalg.det(A))

#b)
g = np.matmul(A,f)                                                                                                #initialize variables
g_measured = np.zeros(len(f))

for i in range(len(f)): #draw g^measured_i
    g_measured[i]= np.random.poisson(lam=g[i])


#c)
A_eigenvalue = np.linalg.eig(A)[0]  #calculate eigenvalues and vec
A_eigenvektor = np.linalg.eig(A)[1]


tmp2 = bubblesort(A_eigenvalue,A_eigenvektor)   #call sorting
A_eig_val_sort = tmp2[0]
A_eig_vec_sort = tmp2[1]

A_eig_vec_sort_inv = np.linalg.inv(A_eig_vec_sort)              #invert eigenvector matrix
D = np.matmul(np.matmul(A_eig_vec_sort_inv,A),A_eig_vec_sort)   #calculate diagonal matrix

D_prime = np.zeros(np.shape(D)) #zeros all entries, which are not on the main diagonal

for i in range(len(D)):
    D_prime[i][i] = D[i][i]
D = np.copy(D_prime)
del D_prime

#d)
c = np.matmul(A_eig_vec_sort_inv,g)#changing of the basis
b = np.matmul(A_eig_vec_sort_inv,f)


tmp = np.matmul(np.linalg.inv(D),A_eig_vec_sort_inv) #calculating b_measured according to the sheet
b_measured = np.matmul(tmp,g_measured)

g_cov = np.diag(g)          #cause g is poisson distributed the cov should be the expected value
b_cov = np.matmul(np.matmul(tmp,g_cov),np.transpose(tmp))
b_norm = b_measured/np.sqrt(np.diag(b_cov))


plt.hist(np.arange(20) - 0.5, bins=np.arange(21) - 0.5, weights=abs(b_norm), histtype="bar")
plt.xticks(np.arange(20))
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel("coefficients")
plt.savefig("build/coeffcients.pdf")
plt.clf()

#e) 
b_reg = np.copy(b_measured)

for i in range(0,len(f)):
    if abs(b_norm[i]) < 1:
        for j in range(i,len(f)):
            cut = j
            b_reg[j] = 0
        break

plt.hist(np.arange(20) - 0.5, bins=np.arange(21) - 0.5, weights=b,          label="Truth",  histtype="step",    color="blue")
plt.hist(np.arange(20) - 0.5, bins=np.arange(21) - 0.5, weights=b_measured, label="$b$",    histtype="step",    color="#0B610B")
plt.hist(np.arange(20) - 0.5, bins=np.arange(21) - 0.5, weights=b_reg,      label="b $reg$",histtype="step",    color="red")
plt.xlim(-0.4,20)
plt.grid()
plt.xlabel("Index")
plt.ylabel("coefficient")
plt.legend(loc="best")
plt.yscale("symlog")
plt.xticks(np.arange(0,20,step=2))
plt.savefig("build/hist.pdf")
plt.clf()

# unfolding
f_unf = np.matmul(A_eig_vec_sort,b_measured)
f_unf_reg = np.matmul(A_eig_vec_sort,b_reg)
f_unf_cov = np.matmul(np.matmul(A_eig_vec_sort,b_cov),np.transpose(A_eig_vec_sort))
f_unf_reg_cov = np.matmul(np.matmul(A_eig_vec_sort,b_reg),np.transpose(A_eig_vec_sort))#you have to regularize the cov matrix somehow 



plt.errorbar(np.arange(20),f_unf,yerr=np.sqrt(np.diag(f_unf_cov)),drawstyle="steps-mid",color="red",label="f unf ",linestyle=('dashed'),capsize=(2))
plt.errorbar(np.arange(20),f_unf_reg,yerr=np.sqrt(f_unf_reg),drawstyle="steps-mid",color="#0B610B",label=r"f_reg",linestyle=('dashdot'),capsize=(2))
plt.plot(np.arange(20),f,drawstyle="steps-mid",color="blue",label=r"Truth")
plt.grid()
plt.xlabel("bins")
plt.ylabel("events")
plt.legend(loc="best")
plt.xticks(np.arange(0,20,step=2))
plt.savefig("build/hist2.pdf")
plt.clf()
