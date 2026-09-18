import numpy as np, turbo_proto as T

def run(K, snr_db, seed, iters=8):
    rng=np.random.default_rng(seed)
    u=rng.integers(0,2,K).astype(np.int8)
    e=T.turbo_encode(u)
    # BPSK map (0->+1,1->-1), add noise, LLR = 2*rx/sigma^2
    sigma2 = 1.0/(2*10**(snr_db/10))     # Es/N0 with unit Es
    def rxllr(bits):
        b=np.asarray(bits,np.int8); x=1-2*b.astype(float)
        y=x+np.sqrt(sigma2)*rng.standard_normal(len(x))
        return 2*y/sigma2
    dec=T.turbo_decode(rxllr(e["sys"]), rxllr(e["par1"]), rxllr(e["par2"]),
                       rxllr(e["tail1_sys"]), rxllr(e["tail2_sys"]), K, iters)
    return np.mean(dec!=u), np.array_equal(dec,u)

print("noiseless round trip (should be exact):")
for K in (40,128,512,1024):
    # very high SNR
    ber,ok=run(K, 30.0, 1)
    print(f"  K={K}: BER={ber:.4f} exact={ok}")

print("\nblock error rate over 40 blocks, K=512:")
for snr in (-2,-1,0,1,2,3):
    errs=sum(0 if run(512,snr,s)[1] else 1 for s in range(40))
    print(f"  {snr:+d}dB: BLER={errs/40:.2f}")

print("\nK=1024 at a few SNR:")
for snr in (-1,0,1,2):
    errs=sum(0 if run(1024,snr,s)[1] else 1 for s in range(30))
    print(f"  {snr:+d}dB: BLER={errs/30:.2f}")
