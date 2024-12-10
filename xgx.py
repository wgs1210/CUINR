import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

paviamat = io.loadmat('/mnt/Data/wgsuser/gitcode/E-NeRV-main/data/pavia/crop2.mat')['truth']#102,512,512
paviamat = paviamat/np.max(paviamat)
#计算具体波段内所有像素值的平均值uc
bandall, h, w = paviamat.shape

allvaluePJ = []
def kjxgx():
    allvalue = []
    for band in range(0, bandall):

        if band % 10 == 0 and band != 0:
            bandImage = paviamat[band,:,:]
            # uc = np.sum(np.mean(bandImage))
            uc = np.sum(bandImage)/(h*w)
            print(uc)
            fenmu = np.sum(np.square(np.abs(bandImage - uc)))

            for w1 in range(0, w):
                for h1 in range(0, h):
                    # print('w=', w1, ' h=', h1, ' w+1=', w1+1, ' h+1=', h1+1)
                    if w1 + 1 < w:#检查是否有下个像素点
                        current_pixel = bandImage[w1, h1]
                        if  h1 + 1 < h:
                            next_pixel = bandImage[w1 + 1, h1 + 1]
                            value = (current_pixel - uc)*(next_pixel - uc)
                            allvalue.append(value)

            fenzi = np.sum(np.array(allvalue))
            # fenzi = np.sum(np.around(np.array(allvalue), decimals=2))
            result = fenzi/fenmu
            allvaluePJ.append(result)
            allvalue = []
            print('band=', band, ' 空间相关性=',result)

    meana = np.mean(np.array(allvaluePJ))
    print('meana=', meana)
#波段相关性。
# bdxgx()

gpy = []
def gpxgx():
    for band in range(0, bandall):
        if band + 1 < bandall:
            bandImage = paviamat[band,:,:]
            bandImagep1 = paviamat[band+1,:,:]

            uc = np.sum(np.mean(bandImage))
            ucp1 = np.sum(np.mean(bandImagep1))

            fenzi = np.sum((bandImage - uc)*(bandImagep1-ucp1))

            fenmu1 = np.sum(np.square(np.abs((bandImage - uc))))
            fenmu2 = np.sum(np.square(np.abs((bandImagep1 - ucp1)))) 
            fenmu = np.sqrt(fenmu1*fenmu2)

            result = fenzi/fenmu
            print('band=', band, ' 普间相关性=', result)
            gpy.append(result)

x = np.arange(0, bandall, 10)
linewidth = 2
plt.plot(x, gpy, linewidth=linewidth, color='y')


#普间相关性
gpxgx()
# kjxgx()



    
