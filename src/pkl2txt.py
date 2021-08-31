import pickle
import numpy as np
from sklearn import metrics

def pkl2txt(di_pkl_name = 'pk1_DI.pkl'):
    output = open(di_pkl_name, 'rb')
    DI = pickle.load(output)
    output.close()
    txt_name = di_pkl_name.split('.')[0]+'.txt'
    file = open(txt_name, mode='w')
    count = 0
    for di in DI:
        str_di = '{0},{1}\n'.format(count, di)
        file.write(str_di)
        count += 1
    file.close()
	
# output = open('X.pkl', 'rb')
# X = pickle.load(output)
# output.close()

# output = open('results.pkl', 'rb')
# results = pickle.load(output)
# output.close()

pkl2txt(di_pkl_name = 'pk1_DI.pkl')
pkl2txt(di_pkl_name = 'pk2_DI.pkl')
pkl2txt(di_pkl_name = 'nopk_DI.pkl')

# output = open('labels.pkl', 'rb')
# labels_ = pickle.load(output)
# output.close()

# print("X -> SihCoe score: %0.3f" % metrics.silhouette_score(X, labels_))