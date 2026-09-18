import numpy as np, turbo_proto as T
def run(K, E, snr_db, seed, rv=0, iters=6):
    rng=np.random.default_rng(seed)
    u=rng.integers(0,2,K).astype(np.int8)
    d0,d1,d2=T.turbo_encode_d(u)
    e=T.rate_match_turbo(d0,d1,d2,E,rv=rv)
    if snr_db>=90:
        ell=(1-2*e.astype(float))*20
    else:
        sigma2=1.0/(2*10**(snr_db/10)); x=1-2*e.astype(float)
        ell=2*(x+np.sqrt(sigma2)*rng.standard_normal(E))/sigma2
    dd0,dd1,dd2=T.rate_dematch_turbo(ell, K+4, E, rv=rv)
    dec=T.turbo_decode_from_d(dd0,dd1,dd2,K,iters)
    return np.array_equal(dec,u)

print("noiseless exact (rate-matched round trip):")
for K,E in [(512,1200),(512,1548),(256,700),(1024,2400)]:
    print(f"  K={K} E={E}: {'OK' if run(K,E,99,1) else 'FAIL'}")

print("\nBLER, K=512:")
for E in (1548, 1200, 900):
    rate=512/E
    for snr in (0,1,2):
        errs=sum(0 if run(512,E,snr,s) else 1 for s in range(15))
        print(f"  E={E} (rate {rate:.2f}) {snr:+d}dB: {errs}/15")
