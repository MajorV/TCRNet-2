import numpy as np
import matplotlib.pyplot as plt

def fas_pd(fas, dets):
    maxv=max(max(fas), max(dets))
    minv=min(min(fas),min(dets))
    step=(maxv-minv)/1000
    print(maxv,minv)
    pds=[]
    fars=[]
    t = minv
    while t < maxv:
        x = np.where(dets > t)
        pd = x[0].shape/ntgt
        pds.append(pd)
        y = np.where(fas>t)
        far = y[0].shape/(nframes * 3.4 * 2.6)
        fars.append(far)
        t += step
    return pds, fars


dets = np.load('./output/dets.npy')
fas = np.load('./output/fas.npy')
ntgt = np.load('./output/ntgt.npy')[0]
nframes = np.load('./output/nframes.npy')[0]

dets2 = np.load('./output/dets2.npy')
fas2 = np.load('./output/fas2.npy')

pds, fars = fas_pd(fas, dets)
pds2, fars2 = fas_pd(fas2, dets2)


# print(len(pds),len(fars))
plt.figure(figsize=(10,8))
plt.plot(fars, pds, label = "Primary TCR Network-2")
plt.plot(fars2, pds2, label = "Primary + Booster TCR Network (boosted)")
plt.xlim(-0.1, 4)
plt.grid()
plt.annotate('%0.2f' % max(pds), xy=(0.9, max(pds)), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % max(pds2), xy=(0.9, max(pds2)), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.legend(loc="lower right")
plt.xlabel("False Alarms per Square Degree", size = 12)
plt.ylabel("Detection Rate", size = 12)
plt.title("Overall ROC at 2500m to 3500m, Day and Night", fontweight="bold", size = 15)
plt.savefig('./output/E2E_roc_2TCRNets_example.png')
plt.show()