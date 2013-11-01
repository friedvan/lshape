#diffusion algorithm for recsys
import time
import random
import numpy as np
##import SplitDataset
##from ior import load, remove
from scipy import sparse
import warnings
warnings.simplefilter('ignore')

####################################################

def item_weight(item, tOU, tUO):
    dU = {}
    dI = {}
    item_degree = 1.0 / len(tOU[item])
    for uid in tOU[item]:
        if uid not in dU:
            dU[uid] = 0.0
        dU[uid] += item_degree
    
    for uid in dU:
        part = float(dU[uid]) / len(tUO[uid])
        for j in tUO[uid]:
            if j not in dI:
                dI[j] = 0.0
            dI[j] += part

    return dI

def diffusion(uid, dUO, dOU, pUO):
    reclist = {}
    for i in W:
        reclist[i] = 0.0
    train = dUO[uid]
    for item in train:
        m = W[item]
        for itemJ in m:
            reclist[itemJ] += m[itemJ]
            

    for item in reclist.keys():
        if item in train or reclist[item] == 0.0:
            del reclist[item]
    return reclist

def addlink_lshape(uo, ou):
    for i in xrange(STEP):
        if len(olist) < 2:
            break
        d, degree, neighbor, o = olist[0]
#        if len(uo)> 76183:
#            print o, 'c, d, n', d, degree, neighbor
        del olist[0]
        if o not in sOU or o not in ou:
            continue
        for u in sOU[o].keys():
            uo.setdefault(u, {})
            uo[u][o] = sUO[u][o]
            ou[o][u] = sUO[u][o]
            del sUO[u][o]
            for oj in sUO[u].keys():
                ou.setdefault(oj, {})
                uo[u][oj] = sUO[u][oj]
                ou[oj][u] = sUO[u][oj]
    return uo, ou


def to_sparse(W, rowlen, collen):
    row, col, data = [], [], []
    for i in W:
        for j in W[i]:
            row.append(i)
            col.append(j)
            data.append(W[i][j])
    newD = sparse.coo_matrix((data, (row, col)), shape=(rowlen+1, collen+1))

    return sparse.csr_matrix(newD)

def random_fill(rec, trainitem):
    reclist = rec.argsort()[0, :-len(rec.data)-1 : -1]
    while len(reclist) < RECLEN:
        ran = np.random.randint(0, len(allitem) - 1)
        if allitem[ran] not in trainitem:
            reclist.append(allitem[ran])
    return reclist

def hitcount(like, reclist):
    hit = 0
    for i in reclist:
        if i in like:
            hit += 1
    return hit


def load_dataset(filename):
    UO, OU = {}, {}
    n = 0
    for line in open(filename):
        n += 1
        if n % 100000 == 0:
            print n
        col = line.rstrip().split(' ')
        uid = int(col[0])
        item = int(col[1])
        time = int(col[2])
                
        if item not in OU:
            OU[item] = {}
        if uid not in UO:
            UO[uid] = {}
        OU[item][uid] = time
        UO[uid][item] = time
        if item not in allitem:
            allitem.append(item)
    print 'load complete'
    return UO, OU

def sortbyneighbor():
##    fout = open('item_neighbor.degree', 'w')
    olist = []
    for o in tOU:
        neighbor = sum([len(tUO[u]) for u in tOU[o]])
        degree = len(tOU[o])
        olist.append((float(degree)/neighbor, degree, neighbor, o))
##        fout.write(' '.join([str(neighbor), str(degree), str(degree/float(neighbor))]) + '\n')

    olist=sorted(olist, key=lambda a:a[0], reverse=False)
    return olist
##    fout.close()
            
############################################
maxuid = 177982 
maxiid = 31446 

allitem = []

STEP = 200 

tUO, tOU = load_dataset('trainset.dat')
pUO, pOU = load_dataset('testset.dat')
sUO, sOU = load_dataset('B.dat')

############
##olist = sorted([(len(tOU[o]), o) for o in tOU], key=lambda a:a[0], reverse=False)
############
print 'sort'
olist = sortbyneighbor()

for i in xrange(50):
    p = []
    recall = [0, 0]
    W = {}

    RECLEN = 20

    t1 = time.clock()

    #calc diffusion matrix
    for item in tOU:
        W[item] = item_weight(item, tOU, tUO)
    t2 = time.clock()
    #to sparse matrix

    R = to_sparse(tUO, maxuid, maxiid)
    D = to_sparse(W, maxiid, maxiid)
    
    t3 = time.clock()
    n = 0    
    for uid in pUO:
        if uid in tUO:
            n += 1
            #diffusion
            rec = R[uid, :].dot(D).toarray()
            for item in tUO[uid]:
                rec[0, item] = 0.0
            if len(rec.data) >= RECLEN:
                reclist = rec.argsort()[0, : -RECLEN - 1 : -1]
            else:
                reclist = random_fill(rec, tUO[uid])
                
            #precision
            hit = hitcount(pUO[uid], reclist)
            cp = float(hit) / RECLEN
            recall[0] += hit
            recall[1] += len(pUO[uid])
            p.append(cp)

    t4=time.clock()
    
    line = ' '.join([str(a) for a in [sum(p)/float(len(p)), float(recall[0]) / recall[1]]])

    tUO, tOU = addlink_lshape(tUO, tOU)
    t5 = time.clock()
#    print t5-t4, t4-t3, t3-t2, t2-t1

    print len(tUO), len(tOU), time.clock() - t1, line



    

