import numpy as np, sib_proto as S, pbch_proto as B
rng=np.random.default_rng(0)
def noisy(sym, snr_db, seed):
    r=np.random.default_rng(seed); sym=np.asarray(sym,complex)
    p=np.mean(np.abs(sym)**2); npow=p/(10**(snr_db/10))
    return sym+np.sqrt(npow/2)*(r.standard_normal(len(sym))+1j*r.standard_normal(len(sym)))

# ---- PCFICH ----
print("PCFICH CFI decode success rate:")
for snr in (20,5,0,-5,-8):
    ok=0
    for s in range(60):
        cfi=1+s%3; nid=(s*7)%504; ns=(s%10)*2
        sym=S.pcfich_encode(cfi,nid,ns)
        got=S.pcfich_decode(noisy(sym,snr,s),nid,ns)
        ok+=(got==cfi)
    print(f"  {snr:+3d}dB: {ok/60:.2f}")

# ---- PDCCH DCI core (single candidate round trip) ----
print("PDCCH DCI decode (AL=4, E=288) success rate:")
for snr in (10,5,0,-3,-6):
    ok=0
    for s in range(60):
        nid=(s*11)%504; ns=(s%10)*2; dci=rng.integers(0,2,28).astype(np.int8)
        scr=S.pdcch_encode(dci, S.SI_RNTI, 288, nid, ns, cce_offset=(s%3)*4)
        rx=noisy(B.qpsk_mod(scr), snr, s)      # QPSK map+demod
        llr=np.empty(288); llr[0::2]=rx.real; llr[1::2]=rx.imag
        # convert soft QPSK back to +/-1-scaled scrambled-bit soft values
        got=S.pdcch_decode_candidate(llr, 44, S.SI_RNTI, nid, ns, cce_offset=(s%3)*4)
        ok+= (got is not None and np.array_equal(got, dci))
    print(f"  {snr:+3d}dB: {ok/60:.2f}")

# wrong RNTI must be rejected
print("Wrong-RNTI rejection:")
fa=0
for s in range(200):
    nid=(s*11)%504; dci=rng.integers(0,2,28).astype(np.int8)
    scr=S.pdcch_encode(dci, S.SI_RNTI, 288, nid, 0, cce_offset=0)
    llr=np.empty(288); rx=B.qpsk_mod(scr); llr[0::2]=rx.real; llr[1::2]=rx.imag
    got=S.pdcch_decode_candidate(llr, 44, 0x1234, nid, 0, cce_offset=0)  # C-RNTI, not SI
    if got is not None: fa+=1
print(f"  decoded under wrong RNTI: {fa}/200")

# false alarm on noise (blind, SI-RNTI)
print("PDCCH false alarm on noise (SI-RNTI blind candidate):")
fa=0
for s in range(2000):
    r=np.random.default_rng(1000+s)
    llr=(r.standard_normal(288))   # noise soft bits
    if S.pdcch_decode_candidate(llr, 44, S.SI_RNTI, 100, 0, cce_offset=0) is not None: fa+=1
print(f"  {fa}/2000")

# blind search finds a placed DCI
print("Blind search over common search space:")
ok=0
for s in range(40):
    nid=(s*13)%504; ns=0; dci=rng.integers(0,2,28).astype(np.int8)
    n_cce=16
    # lay out CCE soft bits (all noise), place DCI at AL8 candidate m=1 -> cce 8
    soft=np.zeros(n_cce*72)
    scr=S.pdcch_encode(dci, S.SI_RNTI, 8*72, nid, ns, cce_offset=8)
    rx=noisy(B.qpsk_mod(scr), 8, s); seg=np.empty(8*72); seg[0::2]=rx.real; seg[1::2]=rx.imag
    soft[8*72:16*72]=seg
    res=S.pdcch_blind_search(soft, 44, S.SI_RNTI, nid, ns, n_cce)
    ok+= (res is not None and np.array_equal(res[0], dci) and res[1]==8 and res[2]==8)
print(f"  found at right (AL,CCE): {ok}/40")
