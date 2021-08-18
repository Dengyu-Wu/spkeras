import pickle
import os

def save_pickle(mdl, name, path):
    pklout = open( os.path.join(path, name + '.pkl'), 'wb')
    pickle.dump(mdl, pklout)
    pklout.close()
    
def load_pickle(path):
    pkl_file = open(path, 'rb')
    mdl = pickle.load(pkl_file)
    pkl_file.close()

    return mdl
