import numpy as np

def create_response_matrix(e,n):
    A = np.eye(n,M=n,k=-1)*e+np.eye(n,M=n,k=0)*(1-e)+np.eye(n,M=n,k=1)*e
    return A
A = create_response_matrix(0.23, 20)
f = np.array([193, 485, 664, 763, 804, 805, 779, 736, 684, 626, 566, 508, 452, 400, 351, 308, 268, 233, 202, 173])
g = np.matmul(A,f)
A_eigenvalue = np.linalg.eig(A)[0]
A_eigenvektor = np.linalg.eig(A)[1]
#print(A_eigenvalue)
#print(A_eigenvektor[0,:])
#print(A_eigenvektor[:,0])
#print(f)
#print(g)
#print(A)
def bubblesort(A,B):
    werte = np.copy(A)
    Matrix = np.copy(B)
    for i in range(len(werte)):
        for j in range(len(werte)-1):
            if werte[j] > werte[j+1]:
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

tmp2 = bubblesort(A_eigenvalue,A_eigenvektor)
A_eig_val_sort = tmp2[0]
A_eig_vec_sort = tmp2[1]
#print(A_eig_val_sort)
#print(A_eig_vec_sort[0,:])
A_eig_vec_sort_inv = np.linalg.inv(A_eig_vec_sort)
D = np.matmul(np.matmul(A_eig_vec_sort_inv,A),A_eig_vec_sort)
