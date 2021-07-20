import numpy as np
import glob
from numpy.linalg import eig as npeig
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from numpy.linalg import inv as npinv


def correlate():
    target_chips = glob.glob('../data/train/chips20x40/targets/' + '*.mat')
    clutter_chips = glob.glob('../data/train/chips20x40/clutter/' + '*.mat')
    d1 = 20
    d2 = 40
    count = len(target_chips)
    print(count, 'target chips')
    alltargets = np.zeros((d1,d2,count))
    for idx,chip in enumerate(target_chips):
        chiparray = loadmat(chip)['target_chip']
        chiparray = chiparray - chiparray.mean()
        alltargets[:,:,idx] = chiparray


    R1 = np.zeros((d1*d2,d1*d2))
    for idx in range(count):
        chipvec = alltargets[:,:,idx].transpose().reshape(d1*d2,1)
        R1 = R1 + np.matmul(chipvec, chipvec.transpose())
    R1 = R1/count
    np.save('./weights_filters/R1',R1)


    count = len(clutter_chips)
    print(count, 'clutter chips')
    allclutter = np.zeros((20,40,count))
    for idx,chip in enumerate(clutter_chips):
        chiparray = loadmat(chip)['clutter_chip']
        chiparray = chiparray - chiparray.mean()
        allclutter[:,:,idx] = chiparray


    x = allclutter[:, :, 0]
    x2 = np.flipud(np.fliplr(x))
    acf = signal.convolve2d(x, x2)
    for idx in range(count-1):
        x = allclutter[:,:,idx + 1]
        x2 = np.flipud(np.fliplr(x))
        tmp = signal.convolve2d(x,x2)
        acf = (acf * idx + tmp) / (idx + 1)

    mask=np.ones((d1,d2));
    pmask=signal.convolve2d(mask,mask,'full')
    cov = acf/pmask

    m = cov.shape[0]
    n = cov.shape[1]

    ad1=int((m+1)/2)
    ad2=int((n+1)/2)
    dim=int(ad1*ad2)


    CM = np.zeros((dim,dim))
    row_index = np.kron(np.ones(ad2), np.arange(0, ad1, 1)).astype("int64")
    col_index = np.kron(np.arange(0, ad2, 1), np.ones(ad1))
    iv = np.column_stack((row_index, col_index))

    for i in range(dim):
        for j in range(dim):
            index = (iv[j, :] - iv[i, :]).astype("int64")
            row = d1 -1 + index[0]
            col = d2 -1 + index[1]
            CM[i, j] = cov[row, col]
            CM[j, i] = CM[i, j]

    R2 = CM
    np.save('./weights_filters/R2',R2)



def make_basis():
    R1 = np.load("./weights_filters/R1.npy")
    R2 = np.load("./weights_filters/R2.npy")
    A = .18 * R1
    B = R2
    S = A + B
    delta, phi = npeig(S)
    sdelta = delta[delta.argsort()]
    sphi = phi[:, delta.argsort()]
    tmp2= np.cumsum(sdelta)/sdelta.sum();
    skip = tmp2[tmp2 < .001].shape[0] - 1
    sdelta = sdelta[skip:]
    sphi = sphi[:,skip:]
    abc = np.matmul(sphi,npinv(sdelta *np.eye(len(sdelta))))
    Sinv = np.matmul(abc,sphi.transpose())
    T=np.matmul(Sinv,(A-B))
    delta, phi = npeig(T)
    delta = np.real(delta)
    phi = np.real(phi)
    sdelta = delta[delta.argsort()]
    sphi = phi[:, delta.argsort()]
    S1=sphi[:,sdelta > .01]
    S2=sphi[:,sdelta < -.01]
    n1 = S1.shape[1]
    n2 = S2.shape[1]
    print(n1,n2)
    np.save('./weights_filters/target_filters',S1)
    np.save('./weights_filters/clutter_filters',S2)



def view_filter(S,idx):
    img_array = S[:, idx].reshape(40, 20).transpose()
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.set_title('ljk', fontsize=10)
    plt.show()


correlate()
make_basis()
target_filters = np.load('./weights_filters/target_filters.npy')
clutter_filters = np.load('./weights_filters/clutter_filters.npy')
print('targets',target_filters.shape)
print('clutter',clutter_filters.shape)

# view_filter(target_filters,-1)
# view_filter(clutter_filters,20)
