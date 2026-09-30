"""FLEX proto validation — round trips on the CONFIRMED machinery (see the STATUS
block in flex_proto.py for what a self-consistent round trip does and does not
prove, and what still needs a real capture)."""
import numpy as np
import flex_proto as F


def test_bch():
    print("1) BCH(31,21)/0x769 round trip + 1- and 2-bit correction")
    rng = np.random.default_rng(1)
    ok_clean = ok_1 = ok_2 = 0
    N = 400
    for _ in range(N):
        info = int(rng.integers(0, 1 << 21))
        cw = F.bch_encode(info)
        d, _, ok = F.bch_decode(cw)
        ok_clean += (ok and d == info)
        # 1-bit error
        e1 = cw ^ (1 << int(rng.integers(0, 31)))
        d1, _, ok1 = F.bch_decode(e1)
        ok_1 += (ok1 and d1 == info)
        # 2-bit error
        i, j = rng.choice(31, 2, replace=False)
        e2 = cw ^ (1 << int(i)) ^ (1 << int(j))
        d2, _, ok2 = F.bch_decode(e2)
        ok_2 += (ok2 and d2 == info)
    print(f"   clean {ok_clean}/{N}  1-bit {ok_1}/{N}  2-bit {ok_2}/{N}")
    assert ok_clean == N and ok_1 == N and ok_2 == N


def test_interleave():
    print("2) 8x32 block interleave/deinterleave identity")
    rng = np.random.default_rng(2)
    for _ in range(200):
        words = [int(rng.integers(0, 1 << 32)) for _ in range(8)]
        bits = F.interleave_block(words)
        back = F.deinterleave_block(bits)
        assert back == words
    print("   200/200 blocks recovered exactly")


def test_fiw():
    print("3) FIW checksum: valid frames parse, corruption is caught")
    good = 0
    for cyc in range(16):
        for frm in (0, 1, 63, 100, 127):
            fiw = F.fiw_build(cyc, frm)
            p = F.fiw_parse(fiw)
            good += (p["valid"] and p["cycle"] == cyc and p["frame"] == frm)
    caught = 0
    rng = np.random.default_rng(3)
    for _ in range(200):
        fiw = F.fiw_build(int(rng.integers(0, 16)), int(rng.integers(0, 128)))
        bad = fiw ^ (1 << int(rng.integers(0, 21)))
        caught += (not F.fiw_parse(bad)["valid"])
    print(f"   valid frames ok: {good}/80 ; single-bit corruption caught: {caught}/200")
    assert good == 80 and caught >= 190      # most single-bit errors change the sum


def test_aln():
    print("4) alphanumeric 3x7-bit word pack/unpack round trip")
    for text in ("EVS BED CLEAN B1532-1", "Bravo Code Yellow ETA 3 min", "A", "12:47 UNIT 7"):
        assert F.aln_unpack(F.aln_pack(text)) == text
    print("   all texts recovered")


def test_end_to_end_2level():
    print("5) 2-level end-to-end: address + alphanumeric vector -> block -> "
          "interleave -> deinterleave -> decode")
    capcode = 1234567
    text = "EVS BED CLEAN B1532-1 ASSIGNED 30 MIN"
    # assemble one block: [address, vector, aln-header, aln message words...]
    aln_words = F.aln_pack(text)
    vector = (F.VEC_ALPHANUMERIC & 0xF) | ((len(aln_words) & 0x7F) << 4)
    header = F.aln_header(msg_num=7)
    infos = [capcode & 0x1FFFFF, vector, header, *aln_words]
    # pad to a whole number of 8-word blocks (a phase is 11 such blocks)
    while len(infos) % F.BLOCK_WORDS:
        infos.append(0)
    # transmit each block through the interleaver and back, then reassemble
    dec = []
    for b0 in range(0, len(infos), F.BLOCK_WORDS):
        block = [F.bch_encode(i) for i in infos[b0:b0 + F.BLOCK_WORDS]]
        words = F.deinterleave_block(F.interleave_block(block))
        for w in words:
            info, _, ok = F.bch_decode(w)
            dec.append(info if ok else None)
    got_capcode = F.capcode_short(dec[0])
    got_type = dec[1] & 0xF
    got_len = (dec[1] >> 4) & 0x7F
    got_text = F.aln_unpack(dec[3:3 + got_len])
    print(f"   capcode {got_capcode}  type {got_type}(=ALN {F.VEC_ALPHANUMERIC})  text {got_text!r}")
    assert got_capcode == capcode
    assert got_type == F.VEC_ALPHANUMERIC
    assert got_text == text


def test_4level_slicer_offset_and_gain_invariant():
    print("6) adaptive 4-level slicer: offset- and gain-invariant level recovery")
    r = np.random.default_rng(11)
    levels = r.integers(0, 4, 3000)
    amp = (2 * levels - 3).astype(float)                       # {-3,-1,1,3}
    sig = amp * 0.4 + 1.1 + 0.08 * r.standard_normal(amp.size)  # gain 0.4, offset 1.1
    rec = F.slice_4level(sig)
    acc = float(np.mean(rec == levels))
    naive = np.zeros(sig.size, np.int8)
    naive[sig >= -2] = 1; naive[sig >= 0] = 2; naive[sig >= 2] = 3
    nacc = float(np.mean(naive == levels))
    print(f"   adaptive {acc:.3f} vs fixed-threshold {nacc:.3f} (offset 1.1, gain 0.4)")
    assert acc > 0.99 and nacc < 0.6


if __name__ == "__main__":
    test_bch()
    test_interleave()
    test_fiw()
    test_aln()
    test_end_to_end_2level()
    test_4level_slicer_offset_and_gain_invariant()
    print("\nALL FLEX PROTO ROUND TRIPS PASSED (self-consistent; see STATUS in "
          "flex_proto.py for the real-capture validation still required)")
