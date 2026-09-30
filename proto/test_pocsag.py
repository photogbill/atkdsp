"""POCSAG proto validation harness (proto-first, like test_turbo.py).

1. noiseless encode->decode round trip must be bit-exact.
2. message success vs SNR: soft-decision (Chase-2) vs hard-slice+hard-BCH
   (the multimon-style path) on IDENTICAL discriminator samples -> the
   coding gain, isolated.
3. detector: P(detect) vs SNR, plus false-alarm rate on pure noise.
"""
import numpy as np
import pocsag_proto as P

FS = 38400.0

MSGS = [
    {"address": 1234568, "function": 3, "kind": "alpha",
     "text": "EMS DISPATCH 5TH AND MAIN CARDIAC"},
    {"address": 979321, "function": 0, "kind": "numeric", "text": "5551234"},
]


def _front_end(messages, baud, snr_db, seed, fdev=4500.0):
    rng = np.random.default_rng(seed)
    x, sps = P.modulate(P.build_stream_bits(messages), FS, baud, fdev)
    if snr_db is not None:
        x = P.add_awgn(x, snr_db, rng)
    disc = P.discriminate(x)
    return P.symbol_soft(disc, sps, P.estimate_phase(disc, sps))


def _match(got, m):
    for g in got:
        if g["address"] != m["address"] or g["function"] != m["function"]:
            continue
        val = g["alpha"] if m["kind"] == "alpha" else g["numeric"][:len(m["text"])]
        if val == m["text"]:
            return True
    return False


def check_exact():
    soft = _front_end(MSGS, 1200, None, 0)
    got = P.decode(soft, use_soft=True)
    ok = all(_match(got, m) for m in MSGS)
    print("1) noiseless round trip")
    for m in MSGS:
        g = next(x for x in got if x["address"] == m["address"])
        val = g["alpha"] if m["kind"] == "alpha" else g["numeric"][:len(m["text"])]
        print(f"   addr={g['address']:>8} f={g['function']} {m['kind']:>7}: {val!r}")
    print(f"   bit-exact: {ok}")
    assert ok, "round trip not exact"


def sweep():
    one = [MSGS[0]]
    print("\n2) alpha message success vs channel SNR (one message, 60 trials)")
    print("   soft = Chase-2 soft-decision; hard = slice + bounded-distance BCH")
    print("   SNR(dB)   soft Chase-2   hard slice+BCH")
    for snr in range(-8, 3, 2):
        s = h = 0
        for seed in range(60):
            soft = _front_end(one, 1200, snr, seed)
            s += _match(P.decode(soft, use_soft=True), one[0])
            h += _match(P.decode(soft, use_soft=False), one[0])
        print(f"   {snr:5d}      {s/60:6.2f}         {h/60:6.2f}")


def codeword_gain():
    print("\n4) per-codeword recovery vs channel SNR (isolates coding gain)")
    print("   SNR(dB)   soft Chase-2   hard BCH")
    N = 30
    for snr in range(-12, 3, 2):
        s_ok = h_ok = tot = 0
        for seed in range(20):
            rng = np.random.default_rng(200 + seed)
            infos = [int(v) for v in rng.integers(0, 1 << 21, N)]
            bits = [1 if i % 2 == 0 else 0 for i in range(64)]
            bits += P._word_bits(P.SYNC)
            for v in infos:
                bits += P._word_bits(P.bch_encode(v))
            x, sps = P.modulate(bits, FS, 1200)
            x = P.add_awgn(x, snr, rng)
            disc = P.discriminate(x)
            soft = P.symbol_soft(disc, sps, P.estimate_phase(disc, sps))
            b = (soft < 0).astype(np.int64)
            sy = P._find_syncs(b, 3)
            if not sy:
                tot += N
                continue
            start, pol = sy[0]
            inv = 1 - b
            for k in range(N):
                i = start + k * 32
                if i + 32 > len(soft):
                    break
                tot += 1
                ih, _, ok_h = P.bch_hard_decode(P._read_word(inv if pol else b, i))
                h_ok += (ok_h and ih == infos[k])
                sf = soft[i:i + 32]
                isf, ok_s = P.bch_soft_decode(-sf if pol else sf)
                s_ok += (ok_s and isf == infos[k])
        print(f"   {snr:5d}      {s_ok/tot:6.3f}         {h_ok/tot:6.3f}")


def detect_roc():
    print("\n3) detector P(detect) vs SNR (40 trials)")
    for snr in (-6, -4, -2, 0, 2, 4):
        d = 0
        for seed in range(40):
            rng = np.random.default_rng(1000 + seed)
            x, _ = P.modulate(P.build_stream_bits(MSGS[:1]), FS, 1200)
            x = P.add_awgn(x, snr, rng)
            r = P.detect(x, FS)
            d += r["present"] and r["baud"] == 1200
        print(f"   {snr:+3d}dB   P(detect)={d/40:.2f}")
    fa = 0
    for seed in range(200):
        rng = np.random.default_rng(5000 + seed)
        noise = rng.standard_normal(40000) + 1j * rng.standard_normal(40000)
        fa += P.detect(noise, FS)["present"]
    print(f"   false alarm on pure noise: {fa}/200")


if __name__ == "__main__":
    check_exact()
    sweep()
    codeword_gain()
    detect_roc()
