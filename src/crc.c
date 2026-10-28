/* The CRC engine and its catalogue (header §17). CRC recovery is in
 * src/crc_recover.c.
 *
 * The register runs MSB-first over message BITS (a bitstream is not
 * byte-aligned), with the Rocksoft parameters: the register is preset to
 * init; with refin each whole input byte is reflected before it goes in;
 * at the end the register is reflected if refout, then XORed with xorout.
 * For message bits m_0..m_{L-1} this computes
 *     reg = (init * x^L + M(x) * x^w) mod g(x),  M(x) = sum m_i x^(L-1-i)
 * which is the identity src/crc_recover.c inverts.
 *
 * Byte-aligned runs go through a 256-entry table built on the stack per call
 * (no global state, header rule 1); short or ragged messages go bitwise. Both
 * paths are the same arithmetic and the tests hold them to each other. */
#include "internal.h"

/* The catalogue: parameter sets are facts, re-typed from the published CRC
 * catalogue (reveng, "Catalogue of parametrised CRC algorithms"); no code was
 * taken. Each check value (the CRC of the nine ASCII bytes "123456789") is
 * asserted by tests/test_bitstream.py. */
typedef struct {
    const char *name;
    int width;
    uint64_t poly, init;
    int refin, refout;
    uint64_t xorout, check;
} crc_entry;

#define F 0
#define T 1
static const crc_entry CATALOG[] = {
    {"CRC-3/GSM",              3, 0x3, 0x0, F, F, 0x7, 0x4},
    {"CRC-3/ROHC",             3, 0x3, 0x7, T, T, 0x0, 0x6},
    {"CRC-4/G-704",            4, 0x3, 0x0, T, T, 0x0, 0x7},
    {"CRC-4/INTERLAKEN",       4, 0x3, 0xf, F, F, 0xf, 0xb},
    {"CRC-5/EPC-C1G2",         5, 0x09, 0x09, F, F, 0x00, 0x00},
    {"CRC-5/G-704",            5, 0x15, 0x00, T, T, 0x00, 0x07},
    {"CRC-5/USB",              5, 0x05, 0x1f, T, T, 0x1f, 0x19},
    {"CRC-6/CDMA2000-A",       6, 0x27, 0x3f, F, F, 0x00, 0x0d},
    {"CRC-6/CDMA2000-B",       6, 0x07, 0x3f, F, F, 0x00, 0x3b},
    {"CRC-6/DARC",             6, 0x19, 0x00, T, T, 0x00, 0x26},
    {"CRC-6/G-704",            6, 0x03, 0x00, T, T, 0x00, 0x06},
    {"CRC-6/GSM",              6, 0x2f, 0x00, F, F, 0x3f, 0x13},
    {"CRC-7/MMC",              7, 0x09, 0x00, F, F, 0x00, 0x75},
    {"CRC-7/ROHC",             7, 0x4f, 0x7f, T, T, 0x00, 0x53},
    {"CRC-7/UMTS",             7, 0x45, 0x00, F, F, 0x00, 0x61},
    {"CRC-8/AUTOSAR",          8, 0x2f, 0xff, F, F, 0xff, 0xdf},
    {"CRC-8/BLUETOOTH",        8, 0xa7, 0x00, T, T, 0x00, 0x26},
    {"CRC-8/CDMA2000",         8, 0x9b, 0xff, F, F, 0x00, 0xda},
    {"CRC-8/DARC",             8, 0x39, 0x00, T, T, 0x00, 0x15},
    {"CRC-8/DVB-S2",           8, 0xd5, 0x00, F, F, 0x00, 0xbc},
    {"CRC-8/GSM-A",            8, 0x1d, 0x00, F, F, 0x00, 0x37},
    {"CRC-8/GSM-B",            8, 0x49, 0x00, F, F, 0xff, 0x94},
    {"CRC-8/HITAG",            8, 0x1d, 0xff, F, F, 0x00, 0xb4},
    {"CRC-8/I-432-1",          8, 0x07, 0x00, F, F, 0x55, 0xa1},
    {"CRC-8/I-CODE",           8, 0x1d, 0xfd, F, F, 0x00, 0x7e},
    {"CRC-8/LTE",              8, 0x9b, 0x00, F, F, 0x00, 0xea},
    {"CRC-8/MAXIM-DOW",        8, 0x31, 0x00, T, T, 0x00, 0xa1},
    {"CRC-8/MIFARE-MAD",       8, 0x1d, 0xc7, F, F, 0x00, 0x99},
    {"CRC-8/NRSC-5",           8, 0x31, 0xff, F, F, 0x00, 0xf7},
    {"CRC-8/OPENSAFETY",       8, 0x2f, 0x00, F, F, 0x00, 0x3e},
    {"CRC-8/ROHC",             8, 0x07, 0xff, T, T, 0x00, 0xd0},
    {"CRC-8/SAE-J1850",        8, 0x1d, 0xff, F, F, 0xff, 0x4b},
    {"CRC-8/SMBUS",            8, 0x07, 0x00, F, F, 0x00, 0xf4},
    {"CRC-8/TECH-3250",        8, 0x1d, 0xff, T, T, 0x00, 0x97},
    {"CRC-8/WCDMA",            8, 0x9b, 0x00, T, T, 0x00, 0x25},
    {"CRC-10/ATM",            10, 0x233, 0x000, F, F, 0x000, 0x199},
    {"CRC-10/CDMA2000",       10, 0x3d9, 0x3ff, F, F, 0x000, 0x233},
    {"CRC-10/GSM",            10, 0x175, 0x000, F, F, 0x3ff, 0x12a},
    {"CRC-11/FLEXRAY",        11, 0x385, 0x01a, F, F, 0x000, 0x5a3},
    {"CRC-11/UMTS",           11, 0x307, 0x000, F, F, 0x000, 0x061},
    {"CRC-12/CDMA2000",       12, 0xf13, 0xfff, F, F, 0x000, 0xd4d},
    {"CRC-12/DECT",           12, 0x80f, 0x000, F, F, 0x000, 0xf5b},
    {"CRC-12/GSM",            12, 0xd31, 0x000, F, F, 0xfff, 0xb34},
    {"CRC-12/UMTS",           12, 0x80f, 0x000, F, T, 0x000, 0xdaf},
    {"CRC-13/BBC",            13, 0x1cf5, 0x0000, F, F, 0x0000, 0x04fa},
    {"CRC-14/DARC",           14, 0x0805, 0x0000, T, T, 0x0000, 0x082d},
    {"CRC-14/GSM",            14, 0x202d, 0x0000, F, F, 0x3fff, 0x30ae},
    {"CRC-15/CAN",            15, 0x4599, 0x0000, F, F, 0x0000, 0x059e},
    {"CRC-15/MPT1327",        15, 0x6815, 0x0000, F, F, 0x0001, 0x2566},
    {"CRC-16/ARC",            16, 0x8005, 0x0000, T, T, 0x0000, 0xbb3d},
    {"CRC-16/CDMA2000",       16, 0xc867, 0xffff, F, F, 0x0000, 0x4c06},
    {"CRC-16/CMS",            16, 0x8005, 0xffff, F, F, 0x0000, 0xaee7},
    {"CRC-16/DDS-110",        16, 0x8005, 0x800d, F, F, 0x0000, 0x9ecf},
    {"CRC-16/DECT-R",         16, 0x0589, 0x0000, F, F, 0x0001, 0x007e},
    {"CRC-16/DECT-X",         16, 0x0589, 0x0000, F, F, 0x0000, 0x007f},
    {"CRC-16/DNP",            16, 0x3d65, 0x0000, T, T, 0xffff, 0xea82},
    {"CRC-16/EN-13757",       16, 0x3d65, 0x0000, F, F, 0xffff, 0xc2b7},
    {"CRC-16/GENIBUS",        16, 0x1021, 0xffff, F, F, 0xffff, 0xd64e},
    {"CRC-16/GSM",            16, 0x1021, 0x0000, F, F, 0xffff, 0xce3c},
    {"CRC-16/IBM-3740",       16, 0x1021, 0xffff, F, F, 0x0000, 0x29b1},
    {"CRC-16/IBM-SDLC",       16, 0x1021, 0xffff, T, T, 0xffff, 0x906e},
    {"CRC-16/ISO-IEC-14443-3-A", 16, 0x1021, 0xc6c6, T, T, 0x0000, 0xbf05},
    {"CRC-16/KERMIT",         16, 0x1021, 0x0000, T, T, 0x0000, 0x2189},
    {"CRC-16/LJ1200",         16, 0x6f63, 0x0000, F, F, 0x0000, 0xbdf4},
    {"CRC-16/M17",            16, 0x5935, 0xffff, F, F, 0x0000, 0x772b},
    {"CRC-16/MAXIM-DOW",      16, 0x8005, 0x0000, T, T, 0xffff, 0x44c2},
    {"CRC-16/MCRF4XX",        16, 0x1021, 0xffff, T, T, 0x0000, 0x6f91},
    {"CRC-16/MODBUS",         16, 0x8005, 0xffff, T, T, 0x0000, 0x4b37},
    {"CRC-16/NRSC-5",         16, 0x080b, 0xffff, T, T, 0x0000, 0xa066},
    {"CRC-16/OPENSAFETY-A",   16, 0x5935, 0x0000, F, F, 0x0000, 0x5d38},
    {"CRC-16/OPENSAFETY-B",   16, 0x755b, 0x0000, F, F, 0x0000, 0x20fe},
    {"CRC-16/PROFIBUS",       16, 0x1dcf, 0xffff, F, F, 0xffff, 0xa819},
    {"CRC-16/RIELLO",         16, 0x1021, 0xb2aa, T, T, 0x0000, 0x63d0},
    {"CRC-16/SPI-FUJITSU",    16, 0x1021, 0x1d0f, F, F, 0x0000, 0xe5cc},
    {"CRC-16/T10-DIF",        16, 0x8bb7, 0x0000, F, F, 0x0000, 0xd0db},
    {"CRC-16/TELEDISK",       16, 0xa097, 0x0000, F, F, 0x0000, 0x0fb3},
    {"CRC-16/TMS37157",       16, 0x1021, 0x89ec, T, T, 0x0000, 0x26b1},
    {"CRC-16/UMTS",           16, 0x8005, 0x0000, F, F, 0x0000, 0xfee8},
    {"CRC-16/USB",            16, 0x8005, 0xffff, T, T, 0xffff, 0xb4c8},
    {"CRC-16/XMODEM",         16, 0x1021, 0x0000, F, F, 0x0000, 0x31c3},
    {"CRC-17/CAN-FD",         17, 0x1685b, 0x00000, F, F, 0x00000, 0x04f03},
    {"CRC-21/CAN-FD",         21, 0x102899, 0x000000, F, F, 0x000000, 0x0ed841},
    {"CRC-24/BLE",            24, 0x00065b, 0x555555, T, T, 0x000000, 0xc25a56},
    {"CRC-24/FLEXRAY-A",      24, 0x5d6dcb, 0xfedcba, F, F, 0x000000, 0x7979bd},
    {"CRC-24/FLEXRAY-B",      24, 0x5d6dcb, 0xabcdef, F, F, 0x000000, 0x1f23b8},
    {"CRC-24/INTERLAKEN",     24, 0x328b63, 0xffffff, F, F, 0xffffff, 0xb4f3e6},
    {"CRC-24/LTE-A",          24, 0x864cfb, 0x000000, F, F, 0x000000, 0xcde703},
    {"CRC-24/LTE-B",          24, 0x800063, 0x000000, F, F, 0x000000, 0x23ef52},
    {"CRC-24/OPENPGP",        24, 0x864cfb, 0xb704ce, F, F, 0x000000, 0x21cf02},
    {"CRC-24/OS-9",           24, 0x800063, 0xffffff, F, F, 0xffffff, 0x200fa5},
    {"CRC-30/CDMA",           30, 0x2030b9c7, 0x3fffffff, F, F, 0x3fffffff, 0x04c34abf},
    {"CRC-31/PHILIPS",        31, 0x04c11db7, 0x7fffffff, F, F, 0x7fffffff, 0x0ce9e46c},
    {"CRC-32/AIXM",           32, 0x814141ab, 0x00000000, F, F, 0x00000000, 0x3010bf7f},
    {"CRC-32/AUTOSAR",        32, 0xf4acfb13, 0xffffffff, T, T, 0xffffffff, 0x1697d06a},
    {"CRC-32/BASE91-D",       32, 0xa833982b, 0xffffffff, T, T, 0xffffffff, 0x87315576},
    {"CRC-32/BZIP2",          32, 0x04c11db7, 0xffffffff, F, F, 0xffffffff, 0xfc891918},
    {"CRC-32/CD-ROM-EDC",     32, 0x8001801b, 0x00000000, T, T, 0x00000000, 0x6ec2edc4},
    {"CRC-32/CKSUM",          32, 0x04c11db7, 0x00000000, F, F, 0xffffffff, 0x765e7680},
    {"CRC-32/ISCSI",          32, 0x1edc6f41, 0xffffffff, T, T, 0xffffffff, 0xe3069283},
    {"CRC-32/ISO-HDLC",       32, 0x04c11db7, 0xffffffff, T, T, 0xffffffff, 0xcbf43926},
    {"CRC-32/JAMCRC",         32, 0x04c11db7, 0xffffffff, T, T, 0x00000000, 0x340bc6d9},
    {"CRC-32/MEF",            32, 0x741b8cd7, 0xffffffff, T, T, 0x00000000, 0xd2c22f51},
    {"CRC-32/MPEG-2",         32, 0x04c11db7, 0xffffffff, F, F, 0x00000000, 0x0376e6e7},
    {"CRC-32/XFER",           32, 0x000000af, 0x00000000, F, F, 0x00000000, 0xbd0be338},
    {"CRC-40/GSM",            40, 0x0004820009ULL, 0x0ULL, F, F, 0xffffffffffULL, 0xd4164fc646ULL},
    {"CRC-64/ECMA-182",       64, 0x42f0e1eba9ea3693ULL, 0x0ULL, F, F, 0x0ULL, 0x6c40df5f0b497347ULL},
    {"CRC-64/GO-ISO",         64, 0x000000000000001bULL, 0xffffffffffffffffULL, T, T,
                                  0xffffffffffffffffULL, 0xb90956c775a41001ULL},
    {"CRC-64/MS",             64, 0x259c84cba6426349ULL, 0xffffffffffffffffULL, T, T,
                                  0x0ULL, 0x75d4b74f024eceeaULL},
    /* CRC-64/NVME (v0.16.0): added to the reveng catalogue after this table
     * was first typed; its check value is asserted like every other. */
    {"CRC-64/NVME",           64, 0xad93d23594c93659ULL, 0xffffffffffffffffULL, T, T,
                              0xffffffffffffffffULL, 0xae8b14860a799888ULL},
    {"CRC-64/REDIS",          64, 0xad93d23594c935a9ULL, 0x0ULL, T, T, 0x0ULL, 0xe9c6d914c4b8d9caULL},
    {"CRC-64/WE",             64, 0x42f0e1eba9ea3693ULL, 0xffffffffffffffffULL, F, F,
                                  0xffffffffffffffffULL, 0x62ec59e3f1a4f00aULL},
    {"CRC-64/XZ",             64, 0x42f0e1eba9ea3693ULL, 0xffffffffffffffffULL, T, T,
                                  0xffffffffffffffffULL, 0x995dc9bbdf1939faULL},
};
#undef F
#undef T
#define NCATALOG ((int)(sizeof CATALOG / sizeof CATALOG[0]))

uint64_t atk_reflect(uint64_t v, int w) {
    uint64_t r = 0;
    for (int i = 0; i < w; ++i) { r = (r << 1) | (v & 1u); v >>= 1; }
    return r;
}

ATK_INLINE uint64_t wmask(int w) { return w >= 64 ? ~0ULL : ((1ULL << w) - 1); }

int atk_crc_valid(const atkdsp_crcspec *s) {
    return s && s->width >= 1 && s->width <= 64
        && (s->poly & ~wmask(s->width)) == 0
        && (s->init & ~wmask(s->width)) == 0
        && (s->xorout & ~wmask(s->width)) == 0;
}

/* The bare register, no reflection of the output and no xorout: init preset,
 * message bits MSB-first (each whole byte reflected first when refin). */
uint64_t atk_crc_register(const uint8_t *bits, size_t nbits, int width, uint64_t poly,
                          uint64_t init, int refin) {
    const uint64_t mask = wmask(width);
    const uint64_t top = 1ULL << (width - 1);
    uint64_t reg = init & mask;
    size_t i = 0;
    const size_t whole = nbits / 8;
    if (width >= 8 && whole >= 32) {
        /* table path: one byte at a time, MSB-first within the (possibly
         * reflected) byte. table[v] = the register change of feeding byte v
         * into the top of a zero register. */
        uint64_t table[256];
        for (int v = 0; v < 256; ++v) {
            uint64_t r = (uint64_t)v << (width - 8);
            for (int k = 0; k < 8; ++k)
                r = (r & top) ? ((r << 1) ^ poly) & mask : (r << 1) & mask;
            table[v] = r;
        }
        for (size_t byte = 0; byte < whole; ++byte) {
            unsigned v = 0;
            const uint8_t *p = bits + 8 * byte;
            if (refin) for (int k = 7; k >= 0; --k) v = (v << 1) | (p[k] & 1u);
            else       for (int k = 0; k < 8; ++k)  v = (v << 1) | (p[k] & 1u);
            const unsigned idx = (unsigned)((reg >> (width - 8)) & 0xffu) ^ v;
            reg = (width == 8 ? 0 : ((reg << 8) & mask)) ^ table[idx];
        }
        i = 8 * whole;
    } else if (refin) {
        for (size_t byte = 0; byte < whole; ++byte) {
            const uint8_t *p = bits + 8 * byte;
            for (int k = 7; k >= 0; --k) {
                const uint64_t fb = ((reg & top) ? 1u : 0u) ^ (p[k] & 1u);
                reg = (reg << 1) & mask;
                if (fb) reg ^= poly;
            }
        }
        i = 8 * whole;
    }
    for (; i < nbits; ++i) {   /* the rest, in the order given */
        const uint64_t fb = ((reg & top) ? 1u : 0u) ^ (bits[i] & 1u);
        reg = (reg << 1) & mask;
        if (fb) reg ^= poly;
    }
    return reg;
}

int atkdsp_crc(const uint8_t *bits, size_t nbits, const atkdsp_crcspec *s, uint64_t *out) {
    if (!atk_crc_valid(s) || !out || (nbits && !bits)) return ATKDSP_E_ARG;
    uint64_t reg = atk_crc_register(bits, nbits, s->width, s->poly, s->init, s->refin);
    if (s->refout) reg = atk_reflect(reg, s->width);
    *out = (reg ^ s->xorout) & wmask(s->width);
    return ATKDSP_OK;
}

int atkdsp_crc_catalog_count(void) { return NCATALOG; }

int atkdsp_crc_catalog(int idx, atkdsp_crcspec *out) {
    if (idx < 0 || idx >= NCATALOG || !out) return ATKDSP_E_ARG;
    const crc_entry *e = &CATALOG[idx];
    memset(out, 0, sizeof *out);
    out->width = e->width; out->poly = e->poly; out->init = e->init;
    out->refin = e->refin; out->refout = e->refout;
    out->xorout = e->xorout; out->check = e->check;
    return ATKDSP_OK;
}

const char *atkdsp_crc_catalog_name(int idx) {
    return (idx < 0 || idx >= NCATALOG) ? NULL : CATALOG[idx].name;
}

/* The CRC of "123456789" under a set — the catalogue's check value. */
uint64_t atk_crc_check_value(const atkdsp_crcspec *s) {
    static const char msg[] = "123456789";
    uint8_t b[72];
    for (int i = 0; i < 9; ++i)
        for (int k = 0; k < 8; ++k) b[8 * i + k] = (uint8_t)(((unsigned char)msg[i] >> (7 - k)) & 1u);
    uint64_t v = 0;
    atkdsp_crc(b, 72, s, &v);
    return v;
}

int atk_crc_catalog_find(const atkdsp_crcspec *s) {
    for (int i = 0; i < NCATALOG; ++i) {
        const crc_entry *e = &CATALOG[i];
        if (e->width == s->width && e->poly == s->poly && e->init == s->init
            && e->refin == s->refin && e->refout == s->refout && e->xorout == s->xorout)
            return i;
    }
    return -1;
}
