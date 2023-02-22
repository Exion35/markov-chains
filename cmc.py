import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
from chain_to_image_functions import image_to_chain, chain_to_image
from skimage.io import imread
from PIL import Image as img

# Question 1

def gauss2(Y,br):
    m1 = bruit(br)[0]
    m2 = bruit(br)[1]
    sig1 = bruit(br)[2]
    sig2 = bruit(br)[3]
    return np.transpose([scs.norm.pdf(Y,m1,sig1),scs.norm.pdf(Y,m2,sig2)])

# Question 2

def forward2(Mat_f,n,A,p10):
    alfa = np.zeros((n,2))
    alfa[0][0] = p10*Mat_f[0][0]
    alfa[0][1] = (1-p10)*Mat_f[0][1]
    alfa[0] = alfa[0]/np.sum(alfa[0])
    for i in range(1,n):
        alfa[i][0] = alfa[i-1][0]*A[0][0]*Mat_f[i][0] + alfa[i-1][1]*A[1][0]*Mat_f[i][0]
        alfa[i][1] = alfa[i-1][0]*A[0][1]*Mat_f[i][1] + alfa[i-1][1]*A[1][1]*Mat_f[i][1]
        alfa[i] = alfa[i]/np.sum(alfa[i])
    return alfa

def forward2_matrix(Mat_f,n,A,p10):
    alfa = np.zeros((n,2))
    alfa[0] = [p10, 1-p10]*Mat_f[0]
    alfa[0] = alfa[0]/np.sum(alfa[0])
    for t in range(1,n):
        alfa[t] = np.dot(alfa[t-1],A)*Mat_f[t]
        alfa[t] = alfa[t]/np.sum(alfa[t])
    return alfa

# Question 3

def backward2(Mat_f,n,A):
    beta = np.zeros((n,2))
    beta[n-1][0], beta[n-1][1] = 0.5, 0.5
    for i in range(n-2,-1,-1):
        beta[i][0] = beta[i+1][0]*A[0][0]*Mat_f[i+1][0] + beta[i+1][1]*A[0][1]*Mat_f[i+1][1]
        beta[i][1] = beta[i+1][0]*A[1][0]*Mat_f[i+1][0] + beta[i+1][1]*A[1][1]*Mat_f[i+1][1]
        beta[i] = beta[i]/np.sum(beta[i])
    return beta

def backward2_matrix(Mat_f,n,A):
    beta = np.zeros((n,2))
    beta[n-1] = [0.5, 0.5]
    for t in range(n-2,-1,-1):
        beta[t] = np.dot(beta[t+1]*Mat_f[t+1],np.transpose(A))
        beta[t] = beta[t]/np.sum(beta[t])
    return beta


# Question 4

def MPM_chaines2(Mat_f,n,cl1,cl2,A,p10):
    alpha = forward2_matrix(Mat_f,n,A,p10)
    beta = backward2_matrix(Mat_f,n,A)
    epsilon = np.transpose(np.transpose(alpha*beta))
    return np.where(epsilon[:,0] > epsilon[:,1], cl1, cl2)

# Question 5

def Seg_chaines_MPM_super2(n,X,cl1,cl2,br):
    p10 = calc_probaprio2(X,cl1,cl2)[0]
    A = calc_transit_prio2(X,n,cl1,cl2)
    Y = bruit_gauss2(X,cl1,cl2,br)
    Mat_f = gauss2(Y,br)
    S = MPM_chaines2(Mat_f,n,cl1,cl2,A,p10)
    plt.plot(X,'-o',label="base")
    plt.plot(Y,'-o',label="bruité")
    plt.plot(S,'o--',c="red",label="segmenté")
    print("Matrice de transition : ",A)
    print("Probabilité p10 : ",p10)
    print("Taux d'erreur : ",taux_erreur(X,S))
    plt.title("Taux d'erreur = " + str(taux_erreur(X,S)))
    plt.legend(loc="upper right")
    plt.show()

def Seg_chaines_MPM_super2_multi(n,cl1,cl2,br):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    axs = [ax1,ax2,ax3,ax4]
    for i in range(4):
        p10 = np.random.rand()
        a = np.random.rand()
        b = 1-a
        c = np.random.rand()
        d = 1-c
        A = [[a,b],[c,d]]
        X = genere_Chaine2(n,cl1,cl2,A,p10)
        Y = bruit_gauss2(X,cl1,cl2,br)
        Mat_f = gauss2(Y,br)
        S = MPM_chaines2(Mat_f,n,cl1,cl2,A,p10)
        axs[i].plot(X,'-o',label="base")
        axs[i].plot(Y,'-o',label="bruité")
        axs[i].plot(S,'o--',c="red",label="segmenté")
        print("Matrice de transition : ",A)
        print("Probabilité p10 : ",p10)
        print("Taux d'erreur : ",taux_erreur(X,S))
    ax1.legend(loc="upper right")
    plt.show()

# Question 7

def calc_transit_prio2(X,n,cl1,cl2):
    a00 = 0
    a0 = 0
    a11 = 0
    a1 = 0
    for k in range(n-1):
        if X[k] == cl1:
            a0 += 1
            if X[k+1] == cl1:
                a00 += 1
        else:
            a1 += 1
            if X[k+1] == cl2:
                a11 += 1
    return np.array([[a00/a0,1-a00/a0],[1-a11/a1,a11/a1]])

# Une application à la segmentation d'images

# Question 4

def Bruitage_image(X_img,br):
    X_ch = image_to_chain(X_img)
    cl1,cl2 = np.unique(X_ch)
    Y = bruit_gauss2(X_ch,cl1,cl2,br)
    return(Y,img.fromarray(chain_to_image(Y)))

def Segmentation_image_MPM(X_img,Y,br):
    X_ch = image_to_chain(X_img)
    n = len(X_ch)
    cl1,cl2 = np.unique(X_ch)
    p10 = calc_probaprio2(X_ch,cl1,cl2)[0]
    print(p10)
    # Y = bruit_gauss2(X_ch,cl1,cl2,br)
    Mat_f = gauss2(Y,br)
    A = calc_transit_prio2(X_ch,n,cl1,cl2)
    S = MPM_chaines2(Mat_f,n,cl1,cl2,A,p10)
    tau = taux_erreur(X_ch,S)
    return(tau, img.fromarray(chain_to_image(S)))

def Segmentation_image_MAP(X_img,Y,br):
    X_ch = image_to_chain(X_img)
    cl1,cl2 = np.unique(X_ch)
    p10 = calc_probaprio2(X_ch,cl1,cl2)[0]
    # Y = bruit_gauss2(X_ch,cl1,cl2,br)
    S = MAP_MPM2(Y,cl1,cl2,p10,br)
    tau = taux_erreur(X_ch,S)
    return(tau, img.fromarray(chain_to_image(S)))

def Segmentation_image_MV(X_img,Y,br):
    X_ch = image_to_chain(X_img)
    cl1,cl2 = np.unique(X_ch)
    # Y = bruit_gauss2(X_ch,cl1,cl2,br)
    S = classif_gauss2(Y,cl1,cl2,br)
    tau = taux_erreur(X_ch,S)
    return(tau, img.fromarray(chain_to_image(S)))

# Fonctions auxiliaires

def bruit(num):
    return{
        1 : [120, 130, 1, 2],
        2 : [127, 127, 1, 5],
        3 : [127, 128, 1, 1],
        4 : [127, 128, 0.1, 0.1],
        5 : [127, 128, 2, 3],
    }[num]

def bruit_gauss2(X,cl1,cl2,br):
    n = len(X)
    m1 = bruit(br)[0]
    m2 = bruit(br)[1]
    sig1 = bruit(br)[2]
    sig2 = bruit(br)[3]
    return((X == cl1)*np.random.normal(m1,sig1,n) + (X == cl2)*np.random.normal(m2,sig2,n))

def classif_gauss2(Y,cl1,cl2,br):
    m1 = bruit(br)[0]
    m2 = bruit(br)[1]
    sig1 = bruit(br)[2]
    sig2 = bruit(br)[3]
    return(np.where(scs.norm.pdf(Y,m1,sig1) > scs.norm.pdf(Y,m2,sig2), cl1, cl2))

def tirage_classe2(p1,cl1,cl2):
    r = np.random.rand()
    if r < p1:
        return cl1
    else:
        return cl2

def genere_Chaine2(n,cl1,cl2,A,p10):
    X = np.zeros((n,))
    X[0] = tirage_classe2(p10,cl1,cl2)
    for k in range(1,n):
        if X[k-1] == cl1:
            X[k] = tirage_classe2(A[0][0],cl1,cl2)
            # p(1->1) = A[0][0] et p(1->2) = A[0][1]
        else:
            X[k] = tirage_classe2(A[1][0],cl1,cl2)
            # p(2->1) = A[1][0] et p(2->2) = A[1][1]
    return X

def taux_erreur(A,B):
    A = np.array(A)
    B = np.array(B)
    boo = (A == B)
    return 1 - np.count_nonzero(boo)/len(A)

def MAP_MPM2(Y,cl1,cl2,p1,br):
    m1 = bruit(br)[0]
    m2 = bruit(br)[1]
    sig1 = bruit(br)[2]
    sig2 = bruit(br)[3]
    return (np.where(p1*scs.norm.pdf(Y,m1,sig1) > (1-p1)*scs.norm.pdf(Y,m2,sig2), cl1, cl2))

def calc_probaprio2(X,cl1,cl2):
    X = np.array(X)
    p1 = np.sum(X == cl1)/len(X)
    p2 = np.sum(X == cl2)/len(X)
    return [p1, p2]

def erreur_moyenneMV(T,X,cl1,cl2,br):
    err = np.zeros((T,))
    for k in range(T):
        Y = bruit_gauss2(X,cl1,cl2,br)
        S = classif_gauss2(Y,cl1,cl2,br)
        err[k] = taux_erreur(X,S)
    return(err.mean())

def erreur_moyenneMAP(T,X,cl1,cl2,br):
    err = np.zeros((T,))
    p1 = calc_probaprio2(X,cl1,cl2)[0]
    for k in range(T):
        Y = bruit_gauss2(X,cl1,cl2,br)
        S = MAP_MPM2(Y,cl1,cl2,p1,br)
        err[k] = taux_erreur(X,S)
    return(err.mean())

def erreur_moyenneMAP_genere(T,A,p10,cl1,cl2,br):
    n = 20
    err = np.zeros((T*T,))
    for k in range(T):
        for i in range(T):
            X = genere_Chaine2(n,cl1,cl2,A,p10)
            Y = bruit_gauss2(X,cl1,cl2,br)
            S = MAP_MPM2(Y,cl1,cl2,p10,br)
            err[k*T+i] = taux_erreur(X,S)
    return(err.mean())

def erreur_moyenneCC(T,A,X,cl1,cl2,br):
    p10 = calc_probaprio2(X,cl1,cl2)[0]
    n = len(X)
    err = np.zeros((T,))
    for k in range(T):
        Y = bruit_gauss2(X,cl1,cl2,br)
        S = MPM_chaines2(gauss2(Y,br),n,cl1,cl2,A,p10)
        err[k] = taux_erreur(X,S)
    return(err.mean())

def erreur_moyenneCC_genere(T,A,p10,cl1,cl2,br):
    n = 20
    err = np.zeros((T*T,))
    for i in range(T*T):
        X = genere_Chaine2(n,cl1,cl2,A,p10)
        Y = bruit_gauss2(X,cl1,cl2,br)
        S = MPM_chaines2(gauss2(Y,br),n,cl1,cl2,A,p10)
        err[i] = taux_erreur(X,S)
    return(err.mean())

if __name__ == "__main__":

    X = np.load("PATH/TO/FILE")
    cl1, cl2 = np.unique(X)
    Y = bruit_gauss2(X,cl1,cl2,1)

    # print(calc_transit_prio2(X,len(X),cl1,cl2))
    # Seg_chaines_MPM_super2(len(X),X,cl1,cl2,1)

    # Mat_f = gauss2(Y,1)
    # print(Mat_f)
    # A = [[0.2,0.8],[0.9,0.1]]

    # print(forward2_matrix(Mat_f,len(Mat_f),A,0.2))
    # print(backward2_matrix(Mat_f,len(Mat_f),A))
    # print(MPM_chaines2(Mat_f,len(Mat_f),cl1,cl2,A,0.2))
    # p10 = 0.98
    # a = 0.73
    # b = 1-a
    # c = 0.15
    # d = 1-c
    # A = np.array([[a,b],[c,d]])
    # print(p10)
    # print(A)
    # for k in range(1,6):
    #     print("Erreur moyenne MAP : ",erreur_moyenneMAP_genere(100,A,p10,100,200,k))
    #     print("Erreur moyenne CC : ",erreur_moyenneCC_genere(100,A,p10,100,200,k))

    X1 = np.load("c:/Users/eloic/Desktop/Travail/TSP/2A/Maths P1 et P2/MAT4501 - Inférence bayésienne dans des modèles markoviens/TP1/signal1.npy")
    cl11, cl21 = np.unique(X1)
    # Seg_chaines_MPM_super2(len(X1),X1,0.5,cl11,cl21,1)

    X2 = np.load("c:/Users/eloic/Desktop/Travail/TSP/2A/Maths P1 et P2/MAT4501 - Inférence bayésienne dans des modèles markoviens/TP1/signal2.npy")
    cl12, cl22 = np.unique(X2)
    # Seg_chaines_MPM_super2(len(X2),X2,0.5,cl12,cl22,1)

    X3 = np.load("c:/Users/eloic/Desktop/Travail/TSP/2A/Maths P1 et P2/MAT4501 - Inférence bayésienne dans des modèles markoviens/TP1/signal3.npy")
    cl13, cl23 = np.unique(X3)
    # Seg_chaines_MPM_super2(len(X3),X3,0.5,cl13,cl23,1)

    X4 = np.load("c:/Users/eloic/Desktop/Travail/TSP/2A/Maths P1 et P2/MAT4501 - Inférence bayésienne dans des modèles markoviens/TP1/signal4.npy")
    cl14, cl24 = np.unique(X4)
    # Seg_chaines_MPM_super2(len(X4),X4,0.5,cl14,cl24,1)

    X5 = np.load("c:/Users/eloic/Desktop/Travail/TSP/2A/Maths P1 et P2/MAT4501 - Inférence bayésienne dans des modèles markoviens/TP1/signal5.npy")
    cl15, cl25 = np.unique(X5)
    # Seg_chaines_MPM_super2(len(X5),X5,0.5,cl15,cl25,1)

    print("Proba a priori (X ) : ",calc_probaprio2(X,cl1,cl2))
    print("Proba a priori (X1) : ",calc_probaprio2(X1,cl11,cl21))
    print("Proba a priori (X2) : ",calc_probaprio2(X2,cl12,cl22))
    print("Proba a priori (X3) : ",calc_probaprio2(X3,cl13,cl23))
    print("Proba a priori (X4) : ",calc_probaprio2(X4,cl14,cl24))
    print("Proba a priori (X5) : ",calc_probaprio2(X5,cl15,cl25))

    for k in range(1,6):
        x = X5
        cla,clb = np.unique(x)
        T = 5
        print("Erreur moyenne MV :",erreur_moyenneMV(T,x,cla,clb,k))
        print("Erreur moyenne MAP : ",erreur_moyenneMAP(T,x,cla,clb,k))
        print("Erreur moyenne CC : ",erreur_moyenneCC(T,calc_transit_prio2(x,len(x),cla,clb),x,cla,clb,k))


    X_img_promenade = imread("img/promenade2.bmp")
    X_img_b = imread("img/beee2.bmp")
    X_img_veau = imread("img/veau2.bmp")
    X_img_zebre = imread("img/zebre2.bmp")
    X_img_cible = imread("img/cible2.bmp")

    Seg_chaines_MPM_super2(len(X2),X2,cl12,cl22,1)
    print(backward2_matrix(gauss2(X,1),len(X),calc_transit_prio2(X,len(X),cl1,cl2)))
    print(backward2(gauss2(bruit_gauss2(X,cl1,cl2,1),1),len(X),calc_transit_prio2(X,len(X),cl1,cl2)))
    print(MPM_chaines2(gauss2(bruit_gauss2(X,cl1,cl2,1),1),len(X),cl1,cl2,calc_transit_prio2(X,len(X),cl1,cl2),calc_probaprio2(X,cl1,cl2)[0]))
    Segmentation_image_MV(X_img_b,1).show()

    fig = plt.figure()
    columns = 4
    rows = 5
    i = 1
    ax = []
    X_img = X_img_cible
    for k in range(1,6):

        Y_ch, Y_img = Bruitage_image(X_img,k)
        fig.add_subplot(rows,columns,i)
        plt.imshow(Y_img)

        err_MV, X_img_MV = Segmentation_image_MV(X_img,Y_ch,k)
        ax.append(fig.add_subplot(rows,columns,i+1))
        ax[-1].set_title(r'$\tau_{erreur} = $' + str(err_MV),fontsize=9)
        ax[-1].label_outer()
        plt.imshow(X_img_MV)

        err_MAP, X_img_MAP = Segmentation_image_MAP(X_img,Y_ch,k)
        ax.append(fig.add_subplot(rows,columns,i+2))
        ax[-1].set_title(r'$\tau_{erreur} = $' + str(err_MAP),fontsize=9)
        ax[-1].label_outer()
        plt.imshow(X_img_MAP)

        err_MPM, X_img_MPM = Segmentation_image_MPM(X_img,Y_ch,k)
        ax.append(fig.add_subplot(rows,columns,i+3))
        ax[-1].set_title(r'$\tau_{erreur} = $' + str(err_MPM),fontsize=9)
        ax[-1].label_outer()
        plt.imshow(X_img_MPM)
        i += 4
    plt.show()
    tabX = [X,X1,X2,X3,X4,X5]
    for k in range(6):
        print(len(tabX[k]))

