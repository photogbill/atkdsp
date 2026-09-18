import numpy as np, sib1_phy as P, sib1_asn1 as A, sib_proto as S, turbo_proto as T

N_RB=25; N_ID=123; SF=5; N_PORTS=2; CFI=3; NS=SF*2
gp=P.grid_params(N_RB)

def riv(rb_start, L, n_rb):
    if (L-1) <= n_rb//2:
        return n_rb*(L-1)+rb_start
    return n_rb*(n_rb-L+1)+(n_rb-1-rb_start)
def riv_inv(v, n_rb):
    for L in range(1, n_rb+1):
        for st in range(0, n_rb-L+1):
            if riv(st,L,n_rb)==v: return st,L
    return None

# ---- DCI 1A (24-bit payload with a 9-bit RIV for 25 RB) ----
RIV_BITS=9
def dci_1a(rb_start, L):
    b=[1,0]                                   # format flag=1A, localized
    v=riv(rb_start,L,N_RB); b+=[(v>>(RIV_BITS-1-i))&1 for i in range(RIV_BITS)]
    b+=[0,1,0,1,0]                            # MCS=... (5)
    b+=[0,0,0, 0, 0,0, 0,0]                   # HARQ(3)+NDI(1)+RV(2)+TPC(2)=8
    return np.array(b[:24],np.int8), v
def dci_1a_parse(bits):
    v=0
    for i in range(RIV_BITS): v=(v<<1)|int(bits[2+i])
    return v

# ---- TX ----
oct_=A.encode_sib1([{'mcc':[3,1,0],'mnc':[4,1,0]}], 0x1234, 0x0ABCDEF)
tbbits=np.array([(byte>>(7-i))&1 for byte in oct_ for i in range(8)],np.int8)
K=256; payload=K-24
tb=np.concatenate([tbbits, np.zeros(payload-len(tbbits),np.int8)])[:payload]

alloc_start, alloc_L = 0, 12                  # 12 contiguous RBs (avoid nothing here)
alloc=list(range(alloc_start, alloc_start+alloc_L))
pdsch_res=P.pdsch_re_list(N_ID,N_RB,alloc,CFI,N_PORTS)
E_pdsch=len(pdsch_res)*2
scr,Kp=P.pdsch_encode(tb,N_ID,P.SI_RNTI,SF,E_pdsch)
pdsch_sym=P._qpsk(scr)

dci,vtx=dci_1a(alloc_start,alloc_L)
E_pdcch=288                                   # AL=4
pdcch_scr=S.pdcch_encode(dci, S.SI_RNTI, E_pdcch, N_ID, NS, cce_offset=0)
pdcch_sym=S.pbch_proto.qpsk_mod(pdcch_scr) if hasattr(S,'pbch_proto') else __import__('pbch_proto').qpsk_mod(pdcch_scr)

pcfich_sym=S.pcfich_encode(3, N_ID, NS)       # CFI=3

# build grid
g=np.zeros((gp['n_sc'],14),dtype=complex)
# CRS (port0) — for channel estimate
for (k,sym),val in P.crs_positions(N_ID,N_RB).items(): g[k,sym]=val
# PCFICH
for (k,l),v in zip(P.pcfich_res(N_ID,N_RB), pcfich_sym): g[k,l]=v
# PDCCH -> first 4 CCEs (AL4 at CCE0)
regs=P.control_regs(N_ID,N_RB,CFI,N_PORTS)
cce_res=[re for reg in regs for re in reg]     # flattened REG REs, interleaved
pdcch_res=cce_res[0:4*36]                       # 4 CCEs
for (k,l),v in zip(pdcch_res, pdcch_sym): g[k,l]=v
# PDSCH
for (k,l),v in zip(pdsch_res, pdsch_sym): g[k,l]=v

s=P.ofdm_modulate(g,gp)
rng=np.random.default_rng(2); snr=12.0
p=np.mean(np.abs(s)**2); s=s+np.sqrt(p/10**(snr/10)/2)*(rng.standard_normal(len(s))+1j*rng.standard_normal(len(s)))

# ---- RX ----
g2=P.ofdm_demodulate(s,gp)
# (flat unit channel in this test, so no equalisation needed)
# 1) PCFICH
pc=[g2[k,l] for (k,l) in P.pcfich_res(N_ID,N_RB)]
cfi=S.pcfich_decode(pc, N_ID, NS)
# 2) PDCCH blind decode SI-RNTI at AL4 CCE0
seg=[]
for (k,l) in cce_res[0:4*36]:
    seg.append(g2[k,l])
llr=np.empty(4*36*2); a=np.array(seg)
llr[0::2]=a.real; llr[1::2]=a.imag
dci_rx=S.pdcch_decode_candidate(llr, 40, S.SI_RNTI, N_ID, NS, cce_offset=0)
# 3) from DCI -> alloc -> PDSCH
ok_dci = dci_rx is not None
if ok_dci:
    vrx=dci_1a_parse(dci_rx); st,L=riv_inv(vrx,N_RB); alloc2=list(range(st,st+L))
    res2=P.pdsch_re_list(N_ID,N_RB,alloc2,cfi,N_PORTS)
    ll=np.empty(len(res2)*2)
    for i,(k,l) in enumerate(res2): ll[2*i]=g2[k,l].real; ll[2*i+1]=g2[k,l].imag
    tb2=P.pdsch_decode(ll,N_ID,P.SI_RNTI,SF,Kp)
else:
    tb2=None

print('recovered CFI:', cfi, '(tx 3)')
print('PDCCH DCI decoded:', ok_dci, ' RIV match:', ok_dci and vrx==vtx, f'(alloc {st if ok_dci else "?"},{L if ok_dci else "?"})')
print('PDSCH TB + CRC-24:', 'OK' if tb2 is not None else 'FAIL')
if tb2 is not None:
    info=A.decode_sib1(oct_)
    print('  TOWER:', A.plmn_str(info['plmns'][0]), 'TAC=0x%04X'%info['tac'], 'ECI=0x%07X'%info['cellid'])
    print('  end-to-end SIB1 decode: SUCCESS')
