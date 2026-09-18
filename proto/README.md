# LTE downlink prototypes (validated Python specs, pending C port)

These are the readable, validated specifications the atkdsp C kernels are (or
will be) held to. Each was validated by an encode->decode round trip and SNR
sweeps before porting. Downlink broadcast only; no subscriber identity.

- pbch_proto.py   — PSS/SSS + PBCH -> MIB (SHIPPED to C: src/lte_pbch.c)
- prach_proto.py  — passive PRACH handset detector (SHIPPED: src/lte_prach.c)
- sib_proto.py    — PCFICH + PDCCH DCI decode core (validated)
- turbo_proto.py  — LTE turbo codec + rate matching (validated)
- sib1_asn1.py    — CRC-24A + SIB1 ASN.1 UPER -> PLMN/TAC/ECI (validated)
- sib1_phy.py     — full-BW grid, OFDM/CRS, PCFICH/PDCCH/PDSCH (validated e2e)
- test_*.py       — the validation harnesses.

SIB1 status: the full chain decodes end to end in Python (subframe ->
operator PLMN / TAC / cell identity). REMAINING before it is trusted live:
1. Confirm the RE geometry (REG numbering, PCFICH/PHICH positions, PSS/SSS
   exclusion in subframe 5) against a REAL capture — a self-consistent round
   trip cannot prove the spec's exact RE choice.
2. TBS table (36.213) + per-bandwidth DCI sizing for a fully blind real decode.
3. Port the chain to C (src/lte_sib1.c) with numpy twins + cross-check tests.
4. Wire into ATK: LteScanner triggers SIB1 -> fills PLMN/ECI/TAC in the LTE tab.
