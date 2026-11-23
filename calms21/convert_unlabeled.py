import sys, os, ijson, numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
f = sys.stdin.buffer
n=0; fr=0
for name, s in ijson.kvitems(f, 'unlabeled_videos', use_float=True):
    kp = np.asarray(s['keypoints'], dtype=np.float32); sc = np.asarray(s['scores'], dtype=np.float32)
    np.savez_compressed(os.path.join(out, name.replace('/', '__')+'.npz'), kp=kp, sc=sc)
    n+=1; fr+=kp.shape[0]
print(n, 'seqs', fr, 'frames', flush=True)
