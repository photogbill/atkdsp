import numpy as np, prach_proto as P

# 1. single preamble: correct root, at various SNR
print("Single preamble, root detection rate over 40 seeds:")
for snr in (20,10,5,0,-5,-10):
    ok=0
    for s in range(40):
        u=1+(s*17)%837
        seq=P.build_capture([(u,0)], snr_db=snr, seed=s)
        d=P.detect(seq)
        ok+= any(x['root']==u for x in d)
    print(f"  {snr:+3d} dB: {ok/40:.2f}")

# 2. robustness to timing offset (window misalignment)
print("Timing-offset robustness (10 dB, root correct):")
for tim in (0,1,5,20,100):
    ok=0
    for s in range(30):
        u=1+(s*29)%837
        seq=P.build_capture([(u,0)], snr_db=10, seed=s, timing=tim)
        d=P.detect(seq)
        ok+= any(x['root']==u for x in d)
    print(f"  roll {tim:4d}: {ok/30:.2f}")

# 3. multiple simultaneous roots detected
print("Multiple concurrent roots (10 dB):")
for nroot in (1,2,3,4):
    ok=0
    for s in range(30):
        roots=[1+((s*13+j*211)%837) for j in range(nroot)]
        roots=list(dict.fromkeys(roots))
        seq=P.build_capture([(u,0) for u in roots], snr_db=10, seed=s)
        d=P.detect(seq)
        found=set(x['root'] for x in d)
        ok+= all(u in found for u in roots)
    print(f"  {nroot} roots: all found in {ok}/30")

# 4. concurrent accesses (same root, different shifts) -> count via PDP
print("Same-root concurrent access counting (15 dB):")
for naccess in (1,2,3):
    good=0
    for s in range(30):
        u=1+(s*7)%837
        shifts=[ (j*137)%839 for j in range(naccess)]
        seq=P.build_capture([(u,sh) for sh in shifts], snr_db=15, seed=s)
        d=P.detect(seq)
        cnt=sum(x['count'] for x in d if x['root']==u)
        good+= (cnt==naccess)
    print(f"  {naccess} accesses: exact count in {good}/30")

# 5. false alarm on pure noise
print("False alarm on noise:")
rng=np.random.default_rng(1); fa=0; trials=1000
for _ in range(trials):
    seq=(rng.standard_normal(839)+1j*rng.standard_normal(839))/np.sqrt(2)
    if P.detect(seq): fa+=1
print(f"  {fa}/{trials} windows flagged a preamble")
