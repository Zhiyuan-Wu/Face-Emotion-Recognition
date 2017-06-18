import numpy as np
import scipy.io as sio
file = open('../data/fer2013/fer2013.csv','r')
counter = -1
Xtr=[]
ytr=[]
Xtepub=[]
ytepub=[]
Xtepri=[]
ytepri=[]
for lines in file:
	counter = counter+1
	if counter==0:
		continue
	else:
		label,pixel,usage = lines.split(',')
		if usage=='Training\n':
			ytr.append(int(label))
			Xtr.append(np.array(map(int,pixel.split(' '))).reshape((48,48)))
		if usage=='PublicTest\n':
			ytepub.append(int(label))
			Xtepub.append(np.array(map(int,pixel.split(' '))).reshape((48,48)))
		if usage=='PrivateTest\n':
			ytepri.append(int(label))
			Xtepri.append(np.array(map(int,pixel.split(' '))).reshape((48,48)))
	if counter%1000==0:
		print counter
sio.savemat('../data/fer2013/fer2013.mat',{'Xtr':np.array(Xtr),'ytr':np.array(ytr).T,'Xtepub':np.array(Xtepub),'ytepub':np.array(ytepub).T,'Xtepri':np.array(Xtepri),'ytepri':np.array(ytepri).T})