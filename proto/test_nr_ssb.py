"""Validation harness for nr_ssb_proto — run from atkdsp/proto:
    python test_nr_ssb.py
Prints one PASS/FAIL line per check and exits non-zero on any FAIL.
Every expected value comes from outside the detector: the spec's own
recurrences, m-sequence identities, the modulator's known placement."""
import sys
import numpy as np
import nr_ssb_proto as P

allok = True


def check(name, ok, note=""):
    global allok
    allok &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{note}]" if note else ""))
    return ok


# ---------------------------------------------------------------- sequences
x0 = P.pss_d(0)
check("PSS: three sequences are one m-sequence shifted by 43 (38.211 7.4.2.2)",
      np.array_equal(P.pss_d(1), np.roll(x0, -43)) and
      np.array_equal(P.pss_d(2), np.roll(x0, -86)))
check("PSS: balanced (sum = -1) and off-peak periodic autocorrelation = -1",
      all(P.pss_d(u).sum() == -1 for u in range(3)) and
      all(abs(np.dot(x0, np.roll(x0, s)) + 1) < 1e-6 for s in range(1, 127)))
xc = max(abs(np.dot(P.pss_d(a), P.pss_d(b)) + 1)
         for a in range(3) for b in range(3) if a != b)
check("PSS: aligned cross-correlation between N_ID2s is the m-sequence's -1 "
      "(they are shifts of one sequence, so only ALIGNED correlation is small)",
      xc < 1e-6)

allseq = {}
for u in range(3):
    for n1 in range(P.N_ID1_MAX):
        allseq[tuple(P.sss_d(n1, u).astype(int))] = (n1, u)
check("SSS: all 1008 (N_ID1, N_ID2) give distinct sequences", len(allseq) == 1008)
tab = P.sss_table(0)
g = tab @ tab.T
np.fill_diagonal(g, 0)
check("SSS: cross-correlation between N_ID1s stays small (aligned, same N_ID2)",
      abs(g).max() < 0.4 * 127, f"max {abs(g).max():.0f} of 127")
check("SSS: values are +-1 products", set(np.unique(tab)) == {-1.0, 1.0})

# ------------------------------------------------------------- resource map
check("map: PSS/SSS on k = 56..182", P.sync_subcarriers()[0] == 56 and
      P.sync_subcarriers()[-1] == 182 and P.sync_subcarriers().size == 127)
check("map: PBCH REs 240 / 96 / 240 in symbols 1 / 2 / 3",
      [P.pbch_subcarriers(l).size for l in (1, 2, 3)] == [240, 96, 240])
check("map: PBCH symbol 2 is k = 0..47 and 192..239 either side of the SSS",
      np.array_equal(P.pbch_subcarriers(2),
                     np.concatenate([np.arange(48), np.arange(192, 240)])))
check("map: DM-RS is every fourth PBCH RE (60 / 24 / 60), k = 4m + PCI mod 4",
      all([P.dmrs_subcarriers(l, 5).size for l in (1, 2, 3)] == [60, 24, 60]
          for _ in [0]) and all((P.dmrs_subcarriers(1, pci) % 4 == pci % 4).all()
                               for pci in (0, 1, 2, 3, 1007)))
check("map: SSB centred on DC (k=120 -> bin 0), 240 bins distinct in 256",
      P.k_to_bin(120) == 0 and len(set(P.k_to_bin(np.arange(240)).tolist())) == 240)

# ---------------------------------------------------------------- timeline
check("time: 14*2^mu symbols = 1 ms, half-frame = 5 ms, at 256*SCS",
      all(P.symbol_start(14 << mu, mu) - P.long_cp(mu) ==
          int(P.sample_rate(15e3 * (1 << mu)) / 1000) for mu in (0, 1)) and
      all(P.half_frame_samples(mu) == int(5 * P.sample_rate(15e3 * (1 << mu)) / 1000)
          for mu in (0, 1)))
check("time: long CP is 20 (15 kHz) and 22 (30 kHz) samples at N=256",
      P.long_cp(0) == 20 and P.long_cp(1) == 22)
check("time: SSB candidates A {2,8,16,22}, B {4,8,16,20}, C {2,8,16,22} (<=3 GHz)",
      P.ssb_symbols("A", 0) == [2, 8, 16, 22] and
      P.ssb_symbols("B", 1) == [4, 8, 16, 20] and
      P.ssb_symbols("C", 1) == [2, 8, 16, 22] and
      len(P.ssb_symbols("C", 1, above_3ghz=True)) == 8)

# ------------------------------------------------------------ end to end
rng = np.random.default_rng(7)
ok = True
notes = []
for mu, case in ((0, "A"), (1, "B"), (1, "C")):
    for _ in range(4):
        pci = int(rng.integers(0, 1008))
        cfo = float(rng.uniform(-5000, 5000))
        off = int(rng.integers(0, 3000))
        x, truth = P.build_stream(pci, mu, case, n_ms=40, snr_db=10,
                                  cfo_hz=cfo, start_offset=off,
                                  seed=int(rng.integers(0, 1 << 30)))
        hits = P.detect(x, mu, threshold=0.3)
        good = [h for h in hits if h.pci == pci]
        exact = sum(1 for h in good if h.offset in truth)
        cfo_err = max((abs(h.cfo_hz - cfo) for h in good), default=1e9)
        this = (len(good) == len(truth) and exact == len(truth) and
                all(h.confirmed for h in good) and cfo_err < 300 and
                all(h.pci == pci for h in hits))
        ok &= this
        if not this:
            notes.append(f"mu{mu}{case} pci{pci}: {len(good)}/{len(truth)} "
                         f"exact {exact} cfo_err {cfo_err:.0f}")
check("e2e: 10 dB, random PCI/CFO/offset, all 3 cases: every SSB found at its "
      "exact sample, right PCI, confirmed, CFO within 300 Hz, no false PCI",
      ok, "; ".join(notes))

# SNR sweep — the published table this harness asserts
print("SNR sweep, mu0 case A, CFO 7 kHz, 40 ms, 4 seeds:")
sweep_ok = True
for snr in (10, 5, 0, -3, -6):
    rows = []
    n_ok = 0
    for seed in range(4):
        pci = 17 + 100 * seed
        x, truth = P.build_stream(pci, 0, "A", n_ms=40, snr_db=snr,
                                  cfo_hz=7000, start_offset=777, seed=seed)
        s = P.summarise([h for h in P.detect(x, 0, threshold=0.3)
                         if h.confirmed])
        top = max(s.values(), key=lambda v: v["metric"]) if s else None
        right = bool(top and top["pci"] == pci)
        n_ok += right
        rows.append(f"{(top['metric'] if top else 0):.2f}/{'ok' if right else 'X'}")
    print(f"  {snr:4d} dB  " + "  ".join(rows))
    if snr >= -3:
        sweep_ok &= n_ok == 4
check("sweep: correct, confirmed PCI at every seed down to -3 dB", sweep_ok)

# negative control
worst = 0.0
n_conf = 0
for mu in (0, 1):
    for seed in range(6):
        r = np.random.default_rng(100 + seed)
        n = int(40e-3 * P.sample_rate(15e3 * (1 << mu)))
        x = (r.standard_normal(n) + 1j * r.standard_normal(n)).astype(np.complex64)
        hits = P.detect(x, mu, threshold=0.0, max_hits=8)
        worst = max(worst, max(h.metric for h in hits))
        n_conf += sum(1 for h in hits if h.confirmed and h.metric > 0.3)
check("negative control: 12 x 40 ms of noise, no confirmed hit above 0.30",
      n_conf == 0, f"noise-only max metric {worst:.3f}; threshold 0.30")
check("negative control: the threshold clears the noise maximum with margin",
      worst < 0.28)

# 20 ms cannot confirm and must say so
x, truth = P.build_stream(300, 0, "A", n_ms=20, snr_db=10, seed=2)
hits = P.detect(x, 0)
check("20 ms look: cell found but reported UNCONFIRMED (period needs 40 ms)",
      len(hits) > 0 and all(h.pci == 300 for h in hits) and
      not any(h.confirmed for h in hits))

# two asynchronous cells sharing the GSCN point, one 6 dB down
xa, _ = P.build_stream(100, 1, "C", n_ms=40, snr_db=60, start_offset=0, seed=3)
xb, _ = P.build_stream(700, 1, "C", n_ms=40, snr_db=60, start_offset=4000,
                       seed=4, amplitude=0.5)
n = min(xa.size, xb.size)
x = xa[:n] + xb[:n]
r = np.random.default_rng(5)
x += (0.1 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)
s = P.summarise([h for h in P.detect(x, 1, threshold=0.3) if h.confirmed])
check("co-channel: two unsynchronised cells, one 6 dB down, both confirmed, "
      "no third", sorted(s) == [100, 700], f"found {sorted(s)}")

# CFO range statement
ok = True
for cfo in (-14000, 14000):
    x, _ = P.build_stream(900, 1, "C", n_ms=40, snr_db=5, cfo_hz=cfo, seed=1)
    s = P.summarise([h for h in P.detect(x, 1, threshold=0.3) if h.confirmed])
    ok &= (900 in s) and abs(s[900]["cfo_hz"] - cfo) < 300
check("CFO: +-14 kHz (about +-0.5 SCS at 30 kHz) found and measured to 300 Hz "
      "at 5 dB", ok)

print("ALL PASS" if allok else "SOME FAILED")
sys.exit(0 if allok else 1)
