import json, sys, numpy as np, os
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
os.makedirs(out, exist_ok=True)
n=0; frames=0
for grp, seqs in d.items():
    for name, s in seqs.items():
        kp = np.asarray(s['keypoints'], dtype=np.float32)  # T,2,2,7
        sc = np.asarray(s['scores'], dtype=np.float32)
        ann = np.asarray(s.get('annotations', []), dtype=np.int8)
        meta = s.get('metadata', {})
        np.savez_compressed(os.path.join(out, name.replace('/', '__') + '.npz'), kp=kp, sc=sc, ann=ann, meta=json.dumps(meta))
        n+=1; frames+=kp.shape[0]
print(src, n, 'seqs', frames, 'frames', kp.shape, sc.shape)
