import numpy as np, pbch_proto as P

def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok

allok = True

# 1. CRC: reference via bit polynomial division
def crc_ref(bits):
    m = list(bits) + [0]*16
    g = [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1]  # D^16..D^0
    m = m[:]
    for i in range(len(bits)):
        if m[i]:
            for j in range(17):
                m[i+j] ^= g[j]
    return np.array(m[-16:], dtype=np.int8)
rng = np.random.default_rng(1)
ok = all(np.array_equal(P.crc16(b), crc_ref(b))
         for b in [rng.integers(0,2,24).astype(np.int8) for _ in range(200)])
allok &= check("crc16 matches polynomial division", ok)

# masks distinguish port counts
ok = True
for npr in (1,2,4):
    mib = rng.integers(0,2,24).astype(np.int8)
    frame = P.crc16_attach(mib, npr)
    ok &= (P.crc16_check(frame) == npr)
allok &= check("crc mask -> port count", ok)

# 2. conv code + viterbi, noiseless
ok = True
for _ in range(50):
    b = rng.integers(0,2,40).astype(np.int8)
    coded = P.conv_encode(b)
    llr = (1 - 2*coded).astype(float) * 10   # clean strong LLRs
    dec = P.viterbi_tb(llr)
    ok &= np.array_equal(dec, b)
allok &= check("tail-biting viterbi noiseless exact", ok)

# 3. rate match round trip (clean)
ok = True
for _ in range(20):
    b = rng.integers(0,2,40).astype(np.int8)
    coded = P.conv_encode(b)
    streams = [coded[0::3], coded[1::3], coded[2::3]]
    e = P.rate_match(streams, 1920)
    ell = (1 - 2*e).astype(float)            # perfect LLRs
    d = P.rate_dematch(ell, 40, seg=(0,1920))
    dec = P.viterbi_tb(d)
    ok &= np.array_equal(dec, b)
allok &= check("rate match/dematch round trip", ok)

# 4. full chain, flat channel, high SNR, all port configs and a few segments
def flat_channels(n_ports, seed):
    r = np.random.default_rng(seed)
    return [ (r.standard_normal()+1j*r.standard_normal())*np.ones(P.NSC)
             for _ in range(n_ports) ]
ok = True
detail = []
for n_ports in (1,2,4):
    for sfn in (0, 1, 2, 3, 100, 511):
        mib,_ = P.mib_pack(50, 0, 1, sfn)
        H = flat_channels(n_ports, 10+sfn+n_ports)
        sig = P.build_pbch_samples(mib, 123, n_ports, sfn, H, snr_db=100)
        info = P.pbch_decode(sig, 123)
        good = info and info['n_ports']==n_ports and info['sfn']==sfn and info['dl_bw']==50
        ok &= bool(good)
        if not good:
            detail.append((n_ports,sfn,info))
allok &= check("full chain noiseless (1/2/4 ports, segments)", ok)
for d in detail[:6]:
    print("   miss:", d)

print("\nALL PASS" if allok else "\nSOME FAILED")
