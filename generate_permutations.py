import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Permutation File Gen')

parser.add_argument('--classes', default=30, type=int,
                    help='Number of permutations to select')
parser.add_argument('--n_grids', default=9, type=int,
                    help='Number of grids')
args = parser.parse_args()

if __name__ == "__main__":
    outname = 'permutations_%d_%d'%(args.n_grids,args.classes)

    P_hat = np.array(list(itertools.permutations(list(range(args.n_grids)), args.n_grids)))
    n = P_hat.shape[0]
    assert args.classes< n ,"Number of classes exceeds total number of permutations"

 
    for i in trange(args.classes):
        if i==0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1,-1])
        else:
            P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)

        P_hat = np.delete(P_hat,j,axis=0)
        D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()

        j = D.argmax()
        
        if i%100==0:
            np.save(outname,P)

    np.save(outname,P)
    print('Created '+outname)
