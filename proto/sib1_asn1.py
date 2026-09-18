"""SIB1 transport block -> tower identity. Downlink broadcast only.

CRC-24A (36.212 5.1.1) and an UNALIGNED PER (UPER) decoder for the part of
BCCH-DL-SCH-Message -> SystemInformationBlockType1 that carries the tower's
public identity: the PLMN list (MCC/MNC = the operator), the trackingAreaCode
(TAC) and the cellIdentity (the 28-bit E-UTRAN cell identity, ECI). It stops
after cellAccessRelatedInfo, which holds all three — nothing about any
subscriber is in SIB1, and none is read.

36.331 ASN.1 (R8 baseline structure). A matching encoder is included so the
decoder is validated by a round trip before it is trusted on a real capture.
"""
import numpy as np


# ---- CRC-24A (36.212 5.1.1) ----------------------------------------------
# gCRC24A = D^24+D^23+D^18+D^17+D^14+D^11+D^10+D^7+D^6+D^5+D^4+D^3+D+1
_G24A = 0x1864CFB           # 25-bit poly (D^24 .. D^0)
def crc24a(bits) -> np.ndarray:
    reg = 0
    for b in bits:
        reg = (reg << 1) | (int(b) & 1)
        if reg & (1 << 24):
            reg ^= _G24A
    for _ in range(24):
        reg <<= 1
        if reg & (1 << 24):
            reg ^= _G24A
    return np.array([(reg >> (23 - i)) & 1 for i in range(24)], dtype=np.int8)


def crc24a_check(bits) -> bool:
    """bits = payload + 24 CRC. True if the CRC checks."""
    return np.array_equal(crc24a(bits[:-24]), np.asarray(bits[-24:]) & 1)


# ---- UPER bit I/O --------------------------------------------------------
class BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, value, nbits):
        for i in range(nbits - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def raw(self, bitlist):
        self.bits.extend(int(b) & 1 for b in bitlist)

    def octets(self):
        b = list(self.bits)
        while len(b) % 8:
            b.append(0)
        return bytes(int("".join(str(x) for x in b[i:i+8]), 2)
                     for i in range(0, len(b), 8))


class BitReader:
    def __init__(self, octets):
        self.bits = [(byte >> (7 - i)) & 1
                     for byte in octets for i in range(8)]
        self.pos = 0

    def u(self, nbits):
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.bits[self.pos]
            self.pos += 1
        return v

    def raw(self, nbits):
        out = self.bits[self.pos:self.pos + nbits]
        self.pos += nbits
        return out


def _nbits(rng):
    """Bits for a constrained value with `rng` = ub-lb+1 possibilities."""
    if rng <= 1:
        return 0
    return (rng - 1).bit_length()


# ---- PLMN identity -------------------------------------------------------
def _enc_plmn(w, mcc, mnc, mcc_present=True, reserved=False):
    # PLMN-Identity SEQUENCE { mcc OPTIONAL, mnc } -> 1 optional bit
    w.u(1 if mcc_present else 0, 1)
    if mcc_present:
        for d in mcc:                       # MCC: 3 digits, 4 bits each
            w.u(d, 4)
    # MNC SIZE(2..3): length (count-2) in 1 bit, then digits
    w.u(len(mnc) - 2, 1)
    for d in mnc:
        w.u(d, 4)
    # (PLMN-IdentityInfo) cellReservedForOperatorUse ENUMERATED{reserved,notReserved}
    w.u(0 if reserved else 1, 1)


def _dec_plmn(r):
    mcc_present = r.u(1)
    mcc = [r.u(4) for _ in range(3)] if mcc_present else None
    mnc_len = r.u(1) + 2
    mnc = [r.u(4) for _ in range(mnc_len)]
    reserved = (r.u(1) == 0)
    return {"mcc": mcc, "mnc": mnc, "reserved": reserved}


# ---- SIB1 (tower-identity portion) ---------------------------------------
def encode_sib1(plmns, tac, cellid, csg_identity=None):
    """Build a BCCH-DL-SCH-Message carrying a SIB1, as octets (for tests).
    `plmns` = list of {mcc:[3], mnc:[2 or 3]}. tac = 16-bit int, cellid =
    28-bit int."""
    w = BitWriter()
    w.u(0, 1)                    # BCCH-DL-SCH-MessageType CHOICE -> c1
    w.u(1, 1)                    # c1 CHOICE -> systemInformationBlockType1
    # SIB1 SEQUENCE: extensible -> ext bit; root optionals p-Max/tdd/nonCrit
    w.u(0, 1)                    # no extensions
    w.u(0, 1)                    # p-Max absent
    w.u(0, 1)                    # tdd-Config absent
    w.u(0, 1)                    # nonCriticalExtension absent
    # cellAccessRelatedInfo SEQUENCE: 1 optional (csg-Identity)
    w.u(1 if csg_identity is not None else 0, 1)
    # plmn-IdentityList SIZE(1..6)
    w.u(len(plmns) - 1, _nbits(6))
    for i, p in enumerate(plmns):
        _enc_plmn(w, p.get("mcc"), p["mnc"],
                  mcc_present=(p.get("mcc") is not None),
                  reserved=p.get("reserved", False))
    w.u(tac, 16)                 # trackingAreaCode BIT STRING(16)
    w.u(cellid, 28)              # cellIdentity BIT STRING(28)
    w.u(0, 1)                    # cellBarred: notBarred (index 1)? see decode
    w.u(0, 1)                    # intraFreqReselection
    w.u(0, 1)                    # csg-Indication BOOLEAN
    if csg_identity is not None:
        w.u(csg_identity, 27)
    return w.octets()


def decode_sib1(octets):
    """Decode the tower identity from a BCCH-DL-SCH-Message. Returns
    {plmns, tac, cellid, ...} or raises on a structural mismatch."""
    r = BitReader(octets)
    if r.u(1) != 0:
        raise ValueError("not c1 (messageClassExtension)")
    if r.u(1) != 1:
        raise ValueError("not SIB1 (systemInformation)")
    ext = r.u(1)                 # SIB1 extension bit
    _pmax = r.u(1); _tdd = r.u(1); _noncrit = r.u(1)   # root optionals
    csg_present = r.u(1)         # cellAccessRelatedInfo optional
    n_plmn = r.u(_nbits(6)) + 1
    plmns = []
    last_mcc = None
    for _ in range(n_plmn):
        p = _dec_plmn(r)
        if p["mcc"] is None:
            p["mcc"] = last_mcc          # inherits the previous PLMN's MCC
        else:
            last_mcc = p["mcc"]
        plmns.append(p)
    tac = r.u(16)
    cellid = r.u(28)
    cell_barred = r.u(1)
    intra = r.u(1)
    csg_ind = r.u(1)
    csg_id = r.u(27) if csg_present else None
    return {"plmns": plmns, "tac": tac, "cellid": cellid,
            "cell_barred": cell_barred, "csg_id": csg_id,
            "extended": bool(ext)}


# ---- pretty helpers ------------------------------------------------------
def plmn_str(p):
    """'MCC-MNC' e.g. '310-410'. MNC keeps its digit count (2 or 3)."""
    mcc = "".join(str(d) for d in p["mcc"])
    mnc = "".join(str(d) for d in p["mnc"])
    return f"{mcc}-{mnc}"
