// fastemu-lf: the same RV32 interpreter as fastemu.c, moved onto the linuxformer memory map
// so the Doom port built for the LX machine (port-lf/) can be played at native speed.
//
// fastemu.c targets the Qwenputer layout: 2 MB of flat RAM at address 0, the WAD in a separate
// DROM window, and the kernel text in a 256 KB ROM at 0x800000. The LX machine is one
// contiguous image at 0x80000000 instead: code below TEXT_END is fetched, everything above it
// (including the 4 MB WAD inside .rodata) is writable data. That difference is why this file
// exists rather than a flag.
//
// It differs from fastemu.c in four ways:
//   1. EVERY address is absolute (0x80000000 + offset). The map is reserved with MAP_NORESERVE
//      so the 2 GB address space costs only the pages the guest actually touches.
//   2. I/O is memory-mapped, not ecall: a byte store to the UART THR prints, a load of the LSR
//      reads ready, and a word store to SYSCON halts (0x5555 success, 0x7777 failure).
//   3. --script plays a key schedule from a file, which is how a scripted playthrough is driven.
//   4. --dump-every dumps a contiguous frame range, which is how a clip is rendered.
//
// Build:  cc -O3 -o fastemu-lf fastemu-lf.c        (no zlib: the PNG writer stores raw blocks)
// Usage:  fastemu-lf doom-lf.elf --wad DOOM.WAD [options]
//
//   --frames N              stop after N rendered frames
//   --steps N               stop after N instructions
//   --hash-at N             print an FNV hash of RAM+regs at frame N and stop
//   --dump a,b,c            dump exactly these frames (max 4096)
//   --dump-every N          dump every Nth frame
//   --dump-from N           first frame --dump-every considers
//   --dump-to N             last frame --dump-every considers
//   --dump-dir DIR          where PPM frames go
//   --live FILE.png         keep re-rendering the current frame to FILE
//   --fbp ADDR              ADDR holds a pointer to the framebuffer (DG_ScreenBuffer)
//   --fb ADDR               ADDR is the framebuffer itself
//   --fc ADDR               qwen_frame_count (watched: a store to it ends a frame)
//   --keys ADDR             qwen_key_state (bit per key)
//   --demo ADDR             qwen_demo_enabled (0 hands the game to the schedule)
//   --script FILE           "frame mask" lines: hold keys from that frame until the next line
//   --control FILE          the old 8-byte live-control file (see fastemu.c)
//   --res WxH               framebuffer size for dumps
//   --guest-out FILE|none   where the guest's UART output goes

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>

#define RAM_BASE  0x80000000u          // the LX machine's one contiguous image
#define TEXT_END  0x40000u             // code below this is fetched; 256 KB
#define RAM_SIZE  0x600000u            // the image: 6 MB (port-lf/doom-lf.ld)
#define MEM_SIZE  (RAM_BASE + RAM_SIZE)
#define STACK_TOP (RAM_BASE + RAM_SIZE - 16)

#define LF_UART_THR 0x10000000u        // a byte store here emits one byte
#define LF_UART_LSR 0x10000005u        // bit 5 = THR empty, bit 6 = transmitter empty
#define LF_SYSCON   0x11100000u        // a word store here halts the machine

static const char *FAULT_NAMES[8] = {"illegal", "ebreak", "misaligned load",
    "misaligned store", "load out of range", "store out of range",
    "bad jump", "ecall"};

static uint8_t *mem;                 // MEM_SIZE bytes, MAP_NORESERVE
static uint32_t regs[32];
static uint32_t entry_pc, pc;
static uint64_t steps;
static int halted, exit_code = -1, fault_cause = -1;
static uint32_t fault_pc;

static uint8_t outbuf[1 << 20];
static size_t outlen;

// PC sampling profiler: --prof FILE writes the pc of every (steps & mask) == 0 instruction, so a
// run can be split by symbol afterwards. Costs one test per instruction when inactive.
static FILE *prof_fp;
static uint64_t prof_mask = 63;

static uint8_t *elf_image;           // the ELF as loaded, kept for .symtab lookups
static size_t elf_size;

static uint32_t rd32(const uint8_t *p) {
    return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
}

static void out_push(uint8_t b) {
    if (outlen < sizeof outbuf) outbuf[outlen++] = b;
}

// ---------------------------------------------------------------- ELF

static int load_elf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { perror("fread"); return -1; }
    fclose(f);
    if (sz < 52 || memcmp(b, "\x7f""ELF", 4) || b[4] != 1) {
        fprintf(stderr, "fastemu-lf: not an ELF32 file\n");
        return -1;
    }
    elf_image = b;
    elf_size = (size_t)sz;
    entry_pc = rd32(b + 0x18);
    uint32_t phoff = rd32(b + 0x1C);
    int phentsize = b[0x2A] | b[0x2B] << 8, phnum = b[0x2C] | b[0x2D] << 8;
    for (int i = 0; i < phnum; i++) {
        const uint8_t *ph = b + phoff + i * phentsize;
        uint32_t type = rd32(ph), off = rd32(ph + 4), va = rd32(ph + 8);
        uint32_t filesz = rd32(ph + 16), memsz = rd32(ph + 20);
        if (type != 1 || !memsz) continue;           // PT_LOAD only
        if (va < RAM_BASE || va + memsz > RAM_BASE + RAM_SIZE) {
            fprintf(stderr, "fastemu-lf: segment at %#x (+%#x) outside the image\n",
                    va, memsz);
            return -1;
        }
        memcpy(mem + va, b + off, filesz);           // bss tail stays zero
    }
    // b is kept: elf_sym reads .symtab out of it
    return 0;
}

// ---------------------------------------------------------------- symbols
// The control cells (framebuffer pointer, frame counter, key mask, demo flag) move whenever the
// binary is relinked, so they are looked up in the ELF's .symtab instead of being hardcoded.

static uint32_t elf_sym(const char *name) {
    if (!elf_image || elf_size < 52) {
        if (getenv("FEDBG")) fprintf(stderr, "elf_sym: no image (img=%p size=%zu)\n",
                                     (void *)elf_image, elf_size);
        return 0;
    }
    const uint8_t *b = elf_image;
    uint32_t shoff = rd32(b + 0x20);
    int shentsize = b[0x2E] | b[0x2F] << 8, shnum = b[0x30] | b[0x31] << 8;
    if (getenv("FEDBG")) fprintf(stderr, "elf_sym: shoff=%#x shentsize=%d shnum=%d look=%s\n",
                                 shoff, shentsize, shnum, name);
    for (int i = 0; i < shnum; i++) {
        const uint8_t *sh = b + shoff + i * shentsize;
        if (rd32(sh + 4) != 2) continue;              // SHT_SYMTAB
        uint32_t off = rd32(sh + 16), size = rd32(sh + 20);
        uint32_t strsh = rd32(sh + 24);
        const uint8_t *strtab = b + rd32(b + shoff + strsh * shentsize + 16);
        for (uint32_t k = 0; k + 16 <= size; k += 16) {
            const uint8_t *s = b + off + k;          // off is a file offset: index from b
            uint32_t nm = rd32(s);
            if (!nm) continue;
            if (!strcmp((const char *)strtab + nm, name)) return rd32(s + 4);
        }
    }
    return 0;
}

// ---------------------------------------------------------------- predecode

enum Op {
    O_ILLEGAL, O_LUI, O_AUIPC, O_JAL, O_JALR,
    O_BEQ, O_BNE, O_BLT, O_BGE, O_BLTU, O_BGEU,
    O_LB, O_LH, O_LW, O_LBU, O_LHU, O_SB, O_SH, O_SW,
    O_ADDI, O_SLTI, O_SLTIU, O_XORI, O_ORI, O_ANDI, O_SLLI, O_SRLI, O_SRAI,
    O_ADD, O_SUB, O_SLL, O_SLT, O_SLTU, O_XOR, O_SRL, O_SRA, O_OR, O_AND,
    O_MUL, O_MULH, O_MULHSU, O_MULHU, O_DIV, O_DIVU, O_REM, O_REMU,
    O_SH1ADD, O_SH2ADD, O_SH3ADD, O_ECALL, O_EBREAK, O_FENCE, O_COUNT
};

typedef struct { uint8_t op, rd, rs1, rs2; int32_t imm; } Ins;
static Ins code[TEXT_END / 4];

static int32_t sx(uint32_t v, int bits) {
    v &= (1u << bits) - 1;
    return (int32_t)(v << (32 - bits)) >> (32 - bits);
}

static void predecode(void) {
    static const uint8_t branch_ops[8] = {O_BEQ, O_BNE, 0, 0, O_BLT, O_BGE, O_BLTU, O_BGEU};
    static const uint8_t load_ops[8] = {O_LB, O_LH, O_LW, 0, O_LBU, O_LHU, 0, 0};
    static const uint8_t store_ops[8] = {O_SB, O_SH, O_SW, 0, 0, 0, 0, 0};
    static const uint8_t imm_ops[8] = {O_ADDI, 0, O_SLTI, O_SLTIU, O_XORI, 0, O_ORI, O_ANDI};
    static const uint8_t r_ops[8] = {O_ADD, O_SLL, O_SLT, O_SLTU, O_XOR, O_SRL, O_OR, O_AND};
    static const uint8_t m_ops[8] = {O_MUL, O_MULH, O_MULHSU, O_MULHU, O_DIV, O_DIVU, O_REM, O_REMU};
    for (uint32_t wi = 0; wi < TEXT_END / 4; wi++) {
        uint32_t w = rd32(mem + RAM_BASE + 4 * wi);
        Ins in = {O_ILLEGAL, 0, 0, 0, 0};
        uint32_t opc = w & 0x7F, f3 = w >> 12 & 7, f7 = w >> 25;
        in.rd = w >> 7 & 31;
        in.rs1 = w >> 15 & 31;
        in.rs2 = w >> 20 & 31;
        int32_t i_imm = sx(w >> 20, 12);
        int32_t s_imm = sx((w >> 25) << 5 | (w >> 7 & 31), 12);
        int32_t b_imm = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 |
                           (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13);
        int32_t j_imm = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 |
                           (w >> 20 & 1) << 11 | (w >> 21 & 0x3FF) << 1, 21);
        switch (opc) {
        case 0x37: in.op = O_LUI;   in.imm = (int32_t)(w & 0xFFFFF000u); break;
        case 0x17: in.op = O_AUIPC; in.imm = (int32_t)(w & 0xFFFFF000u); break;
        case 0x6F: in.op = O_JAL;   in.imm = j_imm; break;
        case 0x67: if (f3 == 0) { in.op = O_JALR; in.imm = i_imm; } break;
        case 0x63: if (branch_ops[f3]) { in.op = branch_ops[f3]; in.imm = b_imm; } break;
        case 0x03: if (load_ops[f3]) { in.op = load_ops[f3]; in.imm = i_imm; } break;
        case 0x23: if (store_ops[f3]) { in.op = store_ops[f3]; in.imm = s_imm; } break;
        case 0x13:
            if (f3 == 1) { if (f7 == 0) { in.op = O_SLLI; in.imm = in.rs2; } }
            else if (f3 == 5) {
                if (f7 == 0) { in.op = O_SRLI; in.imm = in.rs2; }
                else if (f7 == 0x20) { in.op = O_SRAI; in.imm = in.rs2; }
            } else if (imm_ops[f3]) { in.op = imm_ops[f3]; in.imm = i_imm; }
            break;
        case 0x33:
            if (f7 == 0) in.op = r_ops[f3];
            else if (f7 == 0x20 && f3 == 0) in.op = O_SUB;
            else if (f7 == 0x20 && f3 == 5) in.op = O_SRA;
            else if (f7 == 1) in.op = m_ops[f3];
            else if (f7 == 0x10 && f3 == 2) in.op = O_SH1ADD;
            else if (f7 == 0x10 && f3 == 4) in.op = O_SH2ADD;
            else if (f7 == 0x10 && f3 == 6) in.op = O_SH3ADD;
            break;
        case 0x73:
            if (w == 0x00000073u) in.op = O_ECALL;   // none in this binary, kept for safety
            else if (w == 0x00100073u) in.op = O_EBREAK;
            break;                       // CSR forms: none in this binary
        case 0x0F: in.op = O_FENCE; break;
        }
        code[wi] = in;
    }
}

// ---------------------------------------------------------------- frames out

static uint8_t palette[768];
static int have_palette;
static uint32_t fb_addr, fc_addr;     // filled from the ELF's symbols by --fbp/--fc
static int fb_is_ptr;
static uint32_t key_addr, demo_addr;  // demo_addr 0 = not poked
static volatile uint8_t *ctrl;        // mmap'd control file, 8 B
static int resx = 320, resy = 200;

static void poke32(uint32_t addr, uint32_t v) {
    if (addr) memcpy(mem + addr, &v, 4);
}

static void apply_control(void) {
    if (!ctrl) return;
    uint32_t mask;
    memcpy(&mask, (const void *)ctrl, 4);
    uint8_t mode = ctrl[4];
    if (demo_addr) poke32(demo_addr, mode == 0 ? 1u : 0u);
    poke32(key_addr, mode == 1 ? mask : 0u);
}

// The key schedule: each line holds a key mask from its frame until the next line's frame.
// With a schedule the demo is always off, so nothing but the schedule drives the game.
static struct { uint32_t frame, mask; } sched[4096];
static int n_sched;

static void apply_script(uint32_t fc) {
    if (demo_addr) poke32(demo_addr, 0);
    uint32_t mask = 0;
    for (int i = 0; i < n_sched && sched[i].frame <= fc; i++) mask = sched[i].mask;
    poke32(key_addr, mask);
}

static int load_script(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#');          // comments
        if (h) *h = 0;
        h = strchr(line, ';');
        if (h) *h = 0;
        unsigned fr, mask;
        if (sscanf(line, "%u %i", &fr, &mask) != 2 && sscanf(line, "%u 0x%x", &fr, &mask) != 2)
            continue;
        if (n_sched < (int)(sizeof sched / sizeof sched[0])) {
            sched[n_sched].frame = fr;
            sched[n_sched].mask = (uint32_t)mask;
            n_sched++;
        }
    }
    fclose(f);
    return 0;
}

static int load_palette(const char *wadpath) {
    FILE *f = fopen(wadpath, "rb");
    if (!f) { perror(wadpath); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { perror("fread"); return -1; }
    fclose(f);
    uint32_t n = rd32(b + 4), dir = rd32(b + 8);
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *e = b + dir + 16 * i;
        if (!memcmp(e + 8, "PLAYPAL", 8)) {
            memcpy(palette, b + rd32(e), 768);
            have_palette = 1;
            break;
        }
    }
    free(b);
    return have_palette ? 0 : -1;
}

static int fb_base(uint32_t *out) {
    uint32_t base = fb_addr;
    if (fb_is_ptr) {
        base = rd32(mem + fb_addr);
        if (!base) return -1;
    }
    if (base < RAM_BASE || base + (uint32_t)(resx * resy) > RAM_BASE + RAM_SIZE) return -1;
    *out = base;
    return 0;
}

static void put32be(uint8_t *p, uint32_t v) {
    p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
}

// A PNG without zlib: IDAT is a zlib stream made of "stored" deflate blocks, which are a 5-byte
// header and then the raw bytes. Larger than a real deflate, but this build has no zlib and a
// frame is only 192 KB.
static uint32_t adler32_of(const uint8_t *p, size_t n) {
    uint32_t a = 1, b = 0;
    while (n) {
        size_t k = n < 5552 ? n : 5552;      // 5552 keeps b from overflowing before the modulo
        n -= k;
        while (k--) { a += *p++; b += a; }
        a %= 65521; b %= 65521;
    }
    return b << 16 | a;
}

static uint32_t crc32_of(const uint8_t *p, size_t n) {
    static uint32_t tab[256];
    static int init;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            tab[i] = c;
        }
        init = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    while (n--) c = tab[(c ^ *p++) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static size_t deflate_stored(uint8_t *out, const uint8_t *in, size_t n) {
    size_t o = 0;
    out[o++] = 0x78;                         // zlib header: deflate, 32K window
    out[o++] = 0x01;                         // no preset dict, check bits for 0x7801
    size_t left = n;
    do {
        size_t k = left > 65535 ? 65535 : left;
        left -= k;
        out[o++] = left ? 0x00 : 0x01;       // BFINAL on the last block, BTYPE = stored
        out[o++] = k & 0xFF;  out[o++] = k >> 8;
        out[o++] = ~k & 0xFF; out[o++] = (~k >> 8) & 0xFF;
        memcpy(out + o, in + (n - left - k), k);
        o += k;
    } while (left);
    uint32_t ad = adler32_of(in, n);
    out[o++] = ad >> 24; out[o++] = ad >> 16; out[o++] = ad >> 8; out[o++] = ad;
    return o;
}

// One PNG chunk: length, type, data, CRC over type+data.
static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t b4[4];
    put32be(b4, n);
    fwrite(b4, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (n) fwrite(data, 1, n, f);
    uint8_t *buf = malloc(4 + n);
    memcpy(buf, type, 4);
    if (n) memcpy(buf + 4, data, n);
    put32be(b4, crc32_of(buf, 4 + n));
    fwrite(b4, 1, 4, f);
    free(buf);
}

static int write_png(const char *path) {
    int w = resx, h = resy;
    uint32_t base;
    if (fb_base(&base)) return -1;
    const uint8_t *fb = mem + base;
    size_t raw_n = (size_t)h * (1 + 3 * w);
    uint8_t *raw = malloc(raw_n);
    for (int y = 0; y < h; y++) {
        uint8_t *row = raw + y * (1 + 3 * w);
        row[0] = 0;
        for (int x = 0; x < w; x++) {
            const uint8_t *c = palette + 3 * fb[y * w + x];
            row[1 + 3 * x] = c[0];
            row[1 + 3 * x + 1] = c[1];
            row[1 + 3 * x + 2] = c[2];
        }
    }
    uint8_t *z = malloc(raw_n + raw_n / 65535 * 5 + 64);
    size_t zn = deflate_stored(z, raw, raw_n);
    FILE *f = fopen(path, "wb");
    if (!f) { free(raw); free(z); return -1; }
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13];
    put32be(ihdr, w);
    put32be(ihdr + 4, h);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, 13);
    png_chunk(f, "IDAT", z, (uint32_t)zn);
    png_chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(raw); free(z);
    return 0;
}


static int write_ppm(const char *path) {
    uint32_t base;
    if (fb_base(&base)) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", resx, resy);
    const uint8_t *fb = mem + base;
    uint8_t *row = malloc(3 * resx);
    for (int y = 0; y < resy; y++) {
        for (int x = 0; x < resx; x++)
            memcpy(row + 3 * x, palette + 3 * fb[y * resx + x], 3);
        fwrite(row, 1, 3 * resx, f);
    }
    free(row);
    fclose(f);
    return 0;
}

static uint64_t fnv_ram_regs(void) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = RAM_BASE; i < RAM_BASE + RAM_SIZE; i++) {
        h ^= mem[i];
        h *= 1099511628211ULL;
    }
    for (int i = 0; i < 32; i++) {
        uint32_t v = regs[i];
        for (int k = 0; k < 4; k++) { h ^= v >> 8 * k & 0xFF; h *= 1099511628211ULL; }
    }
    for (int k = 0; k < 4; k++) { h ^= pc >> 8 * k & 0xFF; h *= 1099511628211ULL; }
    return h;
}

// ---------------------------------------------------------------- state trace
// A per-frame line of game state, so a key schedule can be tuned against what the game
// actually does instead of against guesses. The field offsets come from the compiled
// structures (offsetof on the host, see work/offsets.c); the addresses come from the ELF.
//   frame leveltime x y angle health kills totalkills nearest_enemy_dist nearest_enemy_health

static uint32_t sym_players, sym_leveltime, sym_totalkills, sym_thinkercap, sym_pmobjthinker;
static uint32_t sym_demoplayback, sym_gamemap, sym_gameepisode;
static const char *trace_path;

static uint32_t ld32(uint32_t a) { uint32_t v; memcpy(&v, mem + a, 4); return v; }

#define THINKER_NEXT 4

// RV32 / ilp32 offsets, printed by port-lf/offsets.c run on this very emulator: on a 64-bit
// host the same structures are laid out differently (8-byte pointers), so these cannot be
// borrowed from an offsetof() on the build machine.
#define MO_X 12
#define MO_Y 16
#define MO_Z 20
#define MO_ANGLE 32
#define MO_TYPE 88
#define MO_FLAGS 104
#define MO_HEALTH 108
#define PL_HEALTH 40
#define PL_KILLCOUNT 212
#define MF_COUNTKILL 4194304u

static void trace_frame(uint32_t fc) {
    static FILE *tf;
    if (!tf) { tf = fopen(trace_path, "w"); if (!tf) return; }
    uint32_t player = sym_players ? sym_players : 0;
    uint32_t leveltime = sym_leveltime ? ld32(sym_leveltime) : 0;
    int32_t kills = player ? (int32_t)ld32(player + PL_KILLCOUNT) : 0;
    int32_t totalkills = sym_totalkills ? (int32_t)ld32(sym_totalkills) : 0;
    uint32_t mo = player ? ld32(player + 0) : 0;
    int32_t px = 0, py = 0, pz = 0, angle = 0, health = 0;
    if (mo) {
        px = (int32_t)ld32(mo + MO_X);
        py = (int32_t)ld32(mo + MO_Y);
        pz = (int32_t)ld32(mo + MO_Z);
        angle = (uint32_t)ld32(mo + MO_ANGLE);
        health = (int32_t)ld32(player + PL_HEALTH);
    }
    // Walk the thinker ring and keep the nearest live killable mobj.
    double best = 1e18;
    int32_t best_health = 0, best_type = -1;
    int32_t best_x = 0, best_y = 0;
    if (sym_thinkercap && sym_pmobjthinker) {
        uint32_t head = sym_thinkercap;
        uint32_t t = ld32(head + THINKER_NEXT);
        for (int guard = 0; guard < 8192 && t && t != head; guard++) {
            uint32_t fn = ld32(t + 8);                    /* thinker.function */
            if (fn == sym_pmobjthinker) {
                uint32_t fl = ld32(t + MO_FLAGS);
                int32_t hp = (int32_t)ld32(t + MO_HEALTH);
                if ((fl & MF_COUNTKILL) && hp > 0) {
                    int32_t ex = (int32_t)ld32(t + MO_X), ey = (int32_t)ld32(t + MO_Y);
                    double dx = (double)ex - px, dy = (double)ey - py;
                    double d = dx * dx + dy * dy;
                    if (d < best) {
                        best = d;
                        best_health = hp;
                        best_type = (int32_t)ld32(t + MO_TYPE);
                        best_x = ex;
                        best_y = ey;
                    }
                }
            }
            t = ld32(t + THINKER_NEXT);
        }
    }
    // Squared map units: the analysis script takes the root, so this file needs no libm.
    double dist = best < 1e17 ? best / (65536.0 * 65536.0) : -1.0;
    double deg = (uint32_t)angle * 360.0 / 4294967296.0;   /* angle_t is binary-angle */
    fprintf(tf, "%u %u %.3f %.3f %.3f %.2f %d %d %d %.2f %d %d %u %u %d %d %d %.1f %.1f\n",
            fc, leveltime, px / 65536.0, py / 65536.0, pz / 65536.0,
            deg, health, kills, totalkills, dist, best_health, best_type,
            demo_addr ? ld32(demo_addr) : 0xFFFFFFFFu,
            key_addr ? ld32(key_addr) : 0xFFFFFFFFu,
            sym_demoplayback ? (int32_t)ld32(sym_demoplayback) : -1,
            sym_gamemap ? (int32_t)ld32(sym_gamemap) : -1,
            sym_gameepisode ? (int32_t)ld32(sym_gameepisode) : -1,
            best_x / 65536.0, best_y / 65536.0);
    fflush(tf);
}

// ---------------------------------------------------------------- autopilot
// Plays the level with closed-loop control and records the key mask it chose for every frame.
// The recording is what matters: the in-weights run has no feedback loop, so it can only replay
// a fixed schedule. The autopilot exists to *find* that schedule, offline, at 900 fps.
//   bit 0 up, 1 down, 2 left, 3 right, 4 fire, 5 use, 6 enter, 7 escape

#define K_UP (1u << 0)
#define K_DOWN (1u << 1)
#define K_LEFT (1u << 2)
#define K_RIGHT (1u << 3)
#define K_FIRE (1u << 4)
#define K_USE (1u << 5)
#define K_ENTER (1u << 6)

static const char *autopilot_out;      // where the recorded schedule is written
static uint32_t *rec_mask;             // mask per frame
static uint32_t rec_n;

static int ap_prev_mask;
static uint32_t ap_fire_until;
static uint32_t ap_level_start;

struct nearest {
    int32_t x, y, health, type;
    double dist;
    int found;
};

static struct nearest nearest_enemy(void) {
    struct nearest n = {0, 0, 0, 0, 0.0, 0};
    uint32_t player = sym_players;
    uint32_t mo = ld32(player + 0);
    if (!mo) return n;
    int32_t px = (int32_t)ld32(mo + MO_X), py = (int32_t)ld32(mo + MO_Y);
    double best = 1e18;
    uint32_t head = sym_thinkercap;
    uint32_t t = ld32(head + THINKER_NEXT);
    for (int guard = 0; guard < 8192 && t && t != head; guard++) {
        if (ld32(t + 8) == sym_pmobjthinker) {
            uint32_t fl = ld32(t + MO_FLAGS);
            int32_t hp = (int32_t)ld32(t + MO_HEALTH);
            if ((fl & MF_COUNTKILL) && hp > 0) {
                int32_t ex = (int32_t)ld32(t + MO_X), ey = (int32_t)ld32(t + MO_Y);
                double dx = (double)ex - px, dy = (double)ey - py;
                double d = dx * dx + dy * dy;
                if (d < best) {
                    best = d;
                    n.x = ex; n.y = ey; n.health = hp;
                    n.type = (int32_t)ld32(t + MO_TYPE);
                    n.found = 1;
                }
            }
        }
        t = ld32(t + THINKER_NEXT);
    }
    if (n.found) n.dist = sqrt(best) / 65536.0;
    return n;
}

static uint32_t autopilot_mask(uint32_t fc) {
    static int32_t last_px, last_py;
    static uint32_t stuck, unstick_until, unstick_flip;
    uint32_t mask = 0;
    uint32_t leveltime = sym_leveltime ? ld32(sym_leveltime) : 0;
    uint32_t mo = sym_players ? ld32(sym_players + 0) : 0;

    if (!mo || leveltime == 0) {
        // Not in a level yet (only without -warp): tap ENTER through the menus.
        return ((fc / 4) % 2) == 0 ? K_ENTER : 0;
    }
    int32_t px = (int32_t)ld32(mo + MO_X), py = (int32_t)ld32(mo + MO_Y);
    uint32_t ang = (uint32_t)ld32(mo + MO_ANGLE);
    struct nearest e = nearest_enemy();
    if (!e.found) return 0;

    // Blocked? A position that stops changing while trying to advance means a wall.
    if (px == last_px && py == last_py) stuck++;
    else stuck = 0;
    last_px = px;
    last_py = py;
    if (stuck > 14 && fc > unstick_until) {
        unstick_until = fc + 24;
        unstick_flip ^= 1;                  // try the other side next time
    }

    double want = atan2((double)(e.y - py), (double)(e.x - px));
    // Doom's angle 0 is east and one turn is 2^32, which is exactly atan2's convention, so the
    // conversion is a plain scale. Adding a whole turn here would flip the steering direction.
    uint32_t want_bam = (uint32_t)(int32_t)(want / 6.283185307179586 * 4294967296.0);
    int32_t err = (int32_t)(want_bam - ang);      // signed, and wraps naturally

    if (fc < unstick_until) {
        // Peel off the wall: back away while turning to the side that is not the aim error,
        // which slides us along the corner instead of pushing into it.
        mask |= K_DOWN;
        mask |= unstick_flip ? K_LEFT : K_RIGHT;
        return mask;
    }

    // The turn quantum is about 4.2 degrees per tic, so a dead band narrower than that never
    // settles: steer only past 8 degrees, and shoot inside 12.
    int32_t abs_err = err < 0 ? -err : err;
    if (abs_err > 95400000) {                             // > 8 deg
        if (err > 0) mask |= K_LEFT;
        else mask |= K_RIGHT;
    }
    if (abs_err < 143100000 && e.dist < 2500.0) {         // < 12 deg: in the gun's spread
        if (ap_fire_until <= fc) ap_fire_until = fc + 8;
        if (fc < ap_fire_until) mask |= K_FIRE;
    }
    if (e.dist > 96.0) mask |= K_UP;
    return mask;
}

static void autopilot_record(uint32_t fc, uint32_t mask) {
    if (!rec_mask) {
        rec_mask = calloc(1u << 20, sizeof *rec_mask);
        if (!rec_mask) return;
    }
    if (fc < (1u << 20)) {
        rec_mask[fc] = mask;
        if (fc + 1 > rec_n) rec_n = fc + 1;
    }
}

static void autopilot_flush(void) {
    if (!autopilot_out || !rec_mask) return;
    FILE *f = fopen(autopilot_out, "w");
    if (!f) return;
    fprintf(f, "# recorded by fastemu-lf --autopilot; frame mask\n");
    fprintf(f, "0 %u\n", rec_mask[0]);
    for (uint32_t i = 1; i < rec_n; i++)
        if (rec_mask[i] != rec_mask[i - 1]) fprintf(f, "%u %u\n", i, rec_mask[i]);
    fclose(f);
}

// ---------------------------------------------------------------- store values from tokens
// With --stores FILE, every store takes its VALUE from the file instead of from the register the
// instruction wrote. The file is the store values in order, extracted from a token stream -- the
// model's predictions. Addresses and control flow still come from executing the program, but the
// bytes that end up in RAM, and therefore every pixel, come from the file. Replaying the
// reference's stream must reproduce the reference bit for bit; that is what makes the same tool
// meaningful when the stream is the model's.
static uint32_t *store_vals;
static size_t n_store_vals, store_pos;

static int load_stores(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    size_t cap = 1 << 20;
    store_vals = malloc(cap * sizeof *store_vals);
    char line[128];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        unsigned long v;
        if (sscanf(line, "%li", &v) != 1) continue;
        if (n_store_vals == cap) {
            cap *= 2;
            store_vals = realloc(store_vals, cap * sizeof *store_vals);
        }
        store_vals[n_store_vals++] = (uint32_t)v;
    }
    fclose(f);
    return 0;
}

// The next value from the feed (0 when the feed is exhausted, so a short feed is visible rather
// than silent).
static uint32_t next_store(void) {
    if (!store_vals) return 0;
    return store_pos < n_store_vals ? store_vals[store_pos++] : 0;
}

// ---------------------------------------------------------------- run config

static uint32_t target_frames, hash_at;
static uint64_t max_steps;
static double pace_fps;                  // >0: hold this wall-clock frame rate
static uint32_t dump_frames[4096];
static int n_dumps;
static uint32_t dump_every, dump_from, dump_to = UINT32_MAX;
static const char *dump_dir, *live_path;
static double run_t0, last_live_write = -1;
static uint32_t last_fc;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int should_dump(uint32_t fc) {
    for (int i = 0; i < n_dumps; i++)
        if (dump_frames[i] == fc) return 1;
    if (dump_every && fc >= dump_from && fc <= dump_to && fc % dump_every == 0) return 1;
    return 0;
}

// Called from the SW handler at the exact instruction where qwen_frame_count changed.
// Returns 1 to stop the run.
static int frame_hook(uint32_t fc) {
    if (autopilot_out) {
        uint32_t m = autopilot_mask(fc);
        autopilot_record(fc, m);
        if (demo_addr) poke32(demo_addr, 0);
        poke32(key_addr, m);
        ap_prev_mask = (int)m;
    }
    else if (n_sched) apply_script(fc);
    else apply_control();
    if (trace_path) trace_frame(fc);
    if (dump_dir && have_palette && should_dump(fc)) {
        char p[1024];
        snprintf(p, sizeof p, "%s/frame-%04u.ppm", dump_dir, fc);
        write_ppm(p);
        fprintf(stderr, "fastemu-lf: dumped frame %u\n", fc);
    }
    if (live_path) {
        double t = now_s();
        if (fc == 1 || t - last_live_write >= 0.015) {
            char tmp[1024];
            snprintf(tmp, sizeof tmp, "%s.tmp", live_path);
            if (write_png(tmp) == 0) rename(tmp, live_path);
            last_live_write = t;
        }
        if (fc % 30 == 0)
            fprintf(stderr, "[%6.1fs] steps=%llu frames=%u (%.1f fps)\n",
                    t - run_t0, (unsigned long long)steps, fc,
                    fc / (t - run_t0));
    }
    if (pace_fps > 0) {
        double target = run_t0 + fc / pace_fps;
        double dt = target - now_s();
        if (dt > 0) {
            struct timespec ts = {(time_t)dt, (long)((dt - (time_t)dt) * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
    if (hash_at && fc == hash_at) {
        printf("hash frame=%u steps=%llu pc=%#x fnv=%016llx\n", fc,
               (unsigned long long)steps, pc,
               (unsigned long long)fnv_ram_regs());
        return 1;
    }
    if (target_frames && fc >= target_frames) return 1;
    return 0;
}

// ---------------------------------------------------------------- the loop

static int in_code(uint32_t t) {
    return !(t & 3) && t >= RAM_BASE && t < RAM_BASE + TEXT_END;
}

static void run(void) {
    static const void *table[O_COUNT] = {
        [O_ILLEGAL] = &&L_ILLEGAL, [O_LUI] = &&L_LUI, [O_AUIPC] = &&L_AUIPC,
        [O_JAL] = &&L_JAL, [O_JALR] = &&L_JALR,
        [O_BEQ] = &&L_BEQ, [O_BNE] = &&L_BNE, [O_BLT] = &&L_BLT,
        [O_BGE] = &&L_BGE, [O_BLTU] = &&L_BLTU, [O_BGEU] = &&L_BGEU,
        [O_LB] = &&L_LB, [O_LH] = &&L_LH, [O_LW] = &&L_LW, [O_LBU] = &&L_LBU,
        [O_LHU] = &&L_LHU, [O_SB] = &&L_SB, [O_SH] = &&L_SH, [O_SW] = &&L_SW,
        [O_ADDI] = &&L_ADDI, [O_SLTI] = &&L_SLTI, [O_SLTIU] = &&L_SLTIU,
        [O_XORI] = &&L_XORI, [O_ORI] = &&L_ORI, [O_ANDI] = &&L_ANDI,
        [O_SLLI] = &&L_SLLI, [O_SRLI] = &&L_SRLI, [O_SRAI] = &&L_SRAI,
        [O_ADD] = &&L_ADD, [O_SUB] = &&L_SUB, [O_SLL] = &&L_SLL,
        [O_SLT] = &&L_SLT, [O_SLTU] = &&L_SLTU, [O_XOR] = &&L_XOR,
        [O_SRL] = &&L_SRL, [O_SRA] = &&L_SRA, [O_OR] = &&L_OR, [O_AND] = &&L_AND,
        [O_MUL] = &&L_MUL, [O_MULH] = &&L_MULH, [O_MULHSU] = &&L_MULHSU,
        [O_MULHU] = &&L_MULHU, [O_DIV] = &&L_DIV, [O_DIVU] = &&L_DIVU,
        [O_REM] = &&L_REM, [O_REMU] = &&L_REMU,
        [O_SH1ADD] = &&L_SH1ADD, [O_SH2ADD] = &&L_SH2ADD, [O_SH3ADD] = &&L_SH3ADD,
        [O_ECALL] = &&L_ECALL, [O_EBREAK] = &&L_EBREAK, [O_FENCE] = &&L_FENCE,
    };
    const Ins *ins;
    int chunk = 8192;
    pc = entry_pc;
    regs[2] = STACK_TOP;                 // __stack_top in port-lf/doom-lf.ld

#define DISPATCH() do { ins = &code[(pc - RAM_BASE) >> 2]; goto *table[ins->op]; } while (0)
#define NEXT() do { steps++; if (prof_fp && !(steps & prof_mask)) { \
    uint32_t _p = pc; fwrite(&_p, 4, 1, prof_fp); } \
    if (--chunk == 0) { chunk = 8192; \
    if (max_steps && steps >= max_steps) goto finish; } DISPATCH(); } while (0)
#define FAULT(c) do { fault_cause = (c); fault_pc = pc; halted = 1; goto finish; } while (0)
#define WR(v) do { if (ins->rd) regs[ins->rd] = (uint32_t)(v); } while (0)

    DISPATCH();

L_LUI:   WR(ins->imm); pc += 4; NEXT();
L_AUIPC: WR(pc + (uint32_t)ins->imm); pc += 4; NEXT();
L_JAL: {
        uint32_t t = pc + (uint32_t)ins->imm;
        if (!in_code(t)) FAULT(6);
        WR(pc + 4); pc = t; NEXT();
    }
L_JALR: {
        uint32_t t = (regs[ins->rs1] + (uint32_t)ins->imm) & ~1u;
        if (!in_code(t)) FAULT(6);
        WR(pc + 4); pc = t; NEXT();
    }
L_BEQ:  if (regs[ins->rs1] == regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BNE:  if (regs[ins->rs1] != regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BLT:  if ((int32_t)regs[ins->rs1] < (int32_t)regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BGE:  if ((int32_t)regs[ins->rs1] >= (int32_t)regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BLTU: if (regs[ins->rs1] < regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BGEU: if (regs[ins->rs1] >= regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_code(t)) FAULT(6); pc = t; } else pc += 4; NEXT();

// The whole image is RAM; the UART and SYSCON sit outside it and are handled here.
#define LOAD_OK(ea, n) ((ea) >= RAM_BASE && (ea) + (n) <= RAM_BASE + RAM_SIZE)
#define EA() (regs[ins->rs1] + (uint32_t)ins->imm)
L_LB: {
        uint32_t ea = EA();
        if (ea == LF_UART_LSR) { WR(0x60); pc += 4; NEXT(); }
        if (!LOAD_OK(ea, 1)) FAULT(4);
        WR((int8_t)mem[ea]); pc += 4; NEXT();
    }
L_LH: {
        uint32_t ea = EA();
        if (ea & 1) FAULT(2);
        if (!LOAD_OK(ea, 2)) FAULT(4);
        int16_t v; memcpy(&v, mem + ea, 2); WR(v); pc += 4; NEXT();
    }
L_LW: {
        uint32_t ea = EA();
        if (ea & 3) FAULT(2);
        if (!LOAD_OK(ea, 4)) FAULT(4);
        uint32_t v; memcpy(&v, mem + ea, 4); WR(v); pc += 4; NEXT();
    }
L_LBU: {
        uint32_t ea = EA();
        if (ea == LF_UART_LSR) { WR(0x60); pc += 4; NEXT(); }
        if (!LOAD_OK(ea, 1)) FAULT(4);
        WR(mem[ea]); pc += 4; NEXT();
    }
L_LHU: {
        uint32_t ea = EA();
        if (ea & 1) FAULT(2);
        if (!LOAD_OK(ea, 2)) FAULT(4);
        uint16_t v; memcpy(&v, mem + ea, 2); WR(v); pc += 4; NEXT();
    }
L_SB: {
        uint32_t ea = EA();
        if (ea == LF_UART_THR) { out_push(regs[ins->rs2] & 0xFF); pc += 4; NEXT(); }
        if (!LOAD_OK(ea, 1)) FAULT(5);
        mem[ea] = (store_vals ? next_store() : regs[ins->rs2]) & 0xFF; pc += 4; NEXT();
    }
L_SH: {
        uint32_t ea = EA();
        if (ea & 1) FAULT(3);
        if (!LOAD_OK(ea, 2)) FAULT(5);
        uint16_t v = store_vals ? (uint16_t)next_store() : (regs[ins->rs2] & 0xFFFF);
        memcpy(mem + ea, &v, 2); pc += 4; NEXT();
    }
L_SW: {
        uint32_t ea = EA();
        if (ea & 3) FAULT(3);
        uint32_t v = store_vals ? next_store() : regs[ins->rs2];
        if (ea == LF_SYSCON) {                 // the machine's halt
            exit_code = v == 0x5555 ? 0 : 1;
            halted = 1;
            steps++;
            goto finish;
        }
        if (!LOAD_OK(ea, 4)) FAULT(5);
        memcpy(mem + ea, &v, 4);
        pc += 4;
        if (ea == fc_addr && v != last_fc) {
            last_fc = v;
            if (frame_hook(v)) goto finish;
        }
        if (--chunk == 0) { chunk = 8192; if (max_steps && steps >= max_steps) goto finish; }
        DISPATCH();
    }

#define A() regs[ins->rs1]
#define B() regs[ins->rs2]
#define IMM() ((uint32_t)ins->imm)
L_ADDI:  WR(A() + IMM()); pc += 4; NEXT();
L_SLTI:  WR((int32_t)A() < ins->imm); pc += 4; NEXT();
L_SLTIU: WR(A() < IMM()); pc += 4; NEXT();
L_XORI:  WR(A() ^ IMM()); pc += 4; NEXT();
L_ORI:   WR(A() | IMM()); pc += 4; NEXT();
L_ANDI:  WR(A() & IMM()); pc += 4; NEXT();
L_SLLI:  WR(A() << (ins->imm & 31)); pc += 4; NEXT();
L_SRLI:  WR(A() >> (ins->imm & 31)); pc += 4; NEXT();
L_SRAI:  WR((int32_t)A() >> (ins->imm & 31)); pc += 4; NEXT();
L_ADD:   WR(A() + B()); pc += 4; NEXT();
L_SUB:   WR(A() - B()); pc += 4; NEXT();
L_SLL:   WR(A() << (B() & 31)); pc += 4; NEXT();
L_SLT:   WR((int32_t)A() < (int32_t)B()); pc += 4; NEXT();
L_SLTU:  WR(A() < B()); pc += 4; NEXT();
L_XOR:   WR(A() ^ B()); pc += 4; NEXT();
L_SRL:   WR(A() >> (B() & 31)); pc += 4; NEXT();
L_SRA:   WR((int32_t)A() >> (B() & 31)); pc += 4; NEXT();
L_OR:    WR(A() | B()); pc += 4; NEXT();
L_AND:   WR(A() & B()); pc += 4; NEXT();
L_MUL:   WR(A() * B()); pc += 4; NEXT();
L_MULH:  WR((int64_t)(int32_t)A() * (int64_t)(int32_t)B() >> 32); pc += 4; NEXT();
L_MULHSU: WR((int64_t)(int32_t)A() * (uint64_t)B() >> 32); pc += 4; NEXT();
L_MULHU: WR((uint64_t)A() * B() >> 32); pc += 4; NEXT();
L_DIV: {
        int32_t a = (int32_t)A(), b = (int32_t)B();
        WR(b == 0 ? 0xFFFFFFFFu : (a == INT32_MIN && b == -1) ? 0x80000000u
                                 : (uint32_t)(a / b));
        pc += 4; NEXT();
    }
L_DIVU:  WR(B() == 0 ? 0xFFFFFFFFu : A() / B()); pc += 4; NEXT();
L_REM: {
        int32_t a = (int32_t)A(), b = (int32_t)B();
        WR(b == 0 ? A() : (a == INT32_MIN && b == -1) ? 0 : (uint32_t)(a % b));
        pc += 4; NEXT();
    }
L_REMU:  WR(B() == 0 ? A() : A() % B()); pc += 4; NEXT();
L_SH1ADD: WR((A() << 1) + B()); pc += 4; NEXT();
L_SH2ADD: WR((A() << 2) + B()); pc += 4; NEXT();
L_SH3ADD: WR((A() << 3) + B()); pc += 4; NEXT();
L_ECALL: {
        uint32_t a7 = regs[17];
        if (a7 == 1) {                 // kept for compatibility; the LX port uses MMIO
            out_push(regs[10] & 0xFF); pc += 4; NEXT();
        }
        if (a7 == 93) { halted = 1; exit_code = (int)regs[10]; goto finish; }
        FAULT(7);
    }
L_EBREAK:  FAULT(1);
L_FENCE:   pc += 4; NEXT();
L_ILLEGAL: FAULT(0);

finish:
    return;
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    const char *elf_path = NULL, *wad_path = NULL, *guest_out = NULL;
    const char *control_path = NULL, *script_path = NULL, *stores_path = NULL;
    const char *sym_fb = NULL, *sym_fc = NULL, *sym_keys = NULL, *sym_demo = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--frames") && i + 1 < argc) target_frames = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--steps") && i + 1 < argc) max_steps = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(a, "--hash-at") && i + 1 < argc) hash_at = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--wad") && i + 1 < argc) wad_path = argv[++i];
        else if (!strcmp(a, "--pace") && i + 1 < argc) pace_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(a, "--live") && i + 1 < argc) live_path = argv[++i];
        else if (!strcmp(a, "--guest-out") && i + 1 < argc) guest_out = argv[++i];
        else if (!strcmp(a, "--fb") && i + 1 < argc) fb_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--fbp") && i + 1 < argc) {
            fb_addr = strtoul(argv[++i], NULL, 0);
            fb_is_ptr = 1;
        }
        else if (!strcmp(a, "--fbp-sym") && i + 1 < argc) { sym_fb = argv[++i]; fb_is_ptr = 1; }
        else if (!strcmp(a, "--fc-sym") && i + 1 < argc) sym_fc = argv[++i];
        else if (!strcmp(a, "--keys-sym") && i + 1 < argc) sym_keys = argv[++i];
        else if (!strcmp(a, "--demo-sym") && i + 1 < argc) sym_demo = argv[++i];
        else if (!strcmp(a, "--fc") && i + 1 < argc) fc_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--keys") && i + 1 < argc) key_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--demo") && i + 1 < argc) demo_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--control") && i + 1 < argc) control_path = argv[++i];
        else if (!strcmp(a, "--script") && i + 1 < argc) script_path = argv[++i];
        else if (!strcmp(a, "--stores") && i + 1 < argc) stores_path = argv[++i];
        else if (!strcmp(a, "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(a, "--prof") && i + 1 < argc) {
            prof_fp = fopen(argv[++i], "wb");
            setvbuf(prof_fp, NULL, _IOFBF, 1 << 20);
            if (getenv("FEPROF_STRIDE"))
                prof_mask = strtoul(getenv("FEPROF_STRIDE"), NULL, 0) - 1;
        }
        else if (!strcmp(a, "--autopilot") && i + 1 < argc) autopilot_out = argv[++i];
        else if (!strcmp(a, "--dump-every") && i + 1 < argc) dump_every = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--dump-from") && i + 1 < argc) dump_from = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--dump-to") && i + 1 < argc) dump_to = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--dump") && i + 1 < argc) {
            char *copy = strdup(argv[++i]), *tok = strtok(copy, ",");
            while (tok && n_dumps < (int)(sizeof dump_frames / sizeof dump_frames[0])) {
                dump_frames[n_dumps++] = strtoul(tok, NULL, 0);
                tok = strtok(NULL, ",");
            }
            free(copy);
        } else if (!strcmp(a, "--res") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &resx, &resy) != 2) {
                fprintf(stderr, "fastemu-lf: bad --res\n");
                return 1;
            }
        } else if (a[0] != '-') elf_path = a;
        else { fprintf(stderr, "fastemu-lf: unknown option %s\n", a); return 1; }
    }
    if (!elf_path) {
        fprintf(stderr, "usage: fastemu-lf <elf> [options]\n");
        return 1;
    }
    if ((n_dumps || dump_every || live_path) && !wad_path) {
        fprintf(stderr, "fastemu-lf: --dump/--live need --wad for the palette\n");
        return 1;
    }
    // The guest addresses 0x80000000.., so the map is 2 GB of *address space* but only the
    // pages the guest touches become resident. MAP_NORESERVE keeps the kernel from charging
    // for all of it.
    mem = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }
    if (load_elf(elf_path)) return 1;
    if (sym_fb)   fb_addr   = elf_sym(sym_fb);
    if (sym_fc)   fc_addr   = elf_sym(sym_fc);
    if (sym_keys) key_addr  = elf_sym(sym_keys);
    if (sym_demo) demo_addr = elf_sym(sym_demo);
    if (trace_path) {          // --trace resolves its own symbols by name
        sym_players     = elf_sym("players");
        sym_leveltime   = elf_sym("leveltime");
        sym_totalkills  = elf_sym("totalkills");
        sym_thinkercap  = elf_sym("thinkercap");
        sym_pmobjthinker = elf_sym("P_MobjThinker");
        sym_demoplayback = elf_sym("demoplayback");
        sym_gamemap = elf_sym("gamemap");
        sym_gameepisode = elf_sym("gameepisode");
        if (!sym_players || !sym_thinkercap || !sym_pmobjthinker) {
            fprintf(stderr, "fastemu-lf: --trace needs players/thinkercap/P_MobjThinker "
                            "in the symbol table (link without stripping)\n");
            return 1;
        }
    }
    if ((sym_fb && !fb_addr) || (sym_fc && !fc_addr) ||
        (sym_keys && !key_addr) || (sym_demo && !demo_addr)) {
        fprintf(stderr, "fastemu-lf: control symbol not found in %s\n", elf_path);
        return 1;
    }
    if (wad_path && load_palette(wad_path)) {
        fprintf(stderr, "fastemu-lf: PLAYPAL not found in %s\n", wad_path);
        return 1;
    }
    if (script_path && load_script(script_path)) return 1;
    if (stores_path && load_stores(stores_path)) return 1;
    predecode();
    if (control_path) {
        int fd = open(control_path, O_RDWR | O_CREAT, 0644);
        if (fd < 0) { perror(control_path); return 1; }
        if (ftruncate(fd, 8)) { perror("ftruncate"); return 1; }
        void *p = mmap(NULL, 8, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { perror("mmap"); return 1; }
        ctrl = (volatile uint8_t *)p;
        close(fd);
    }
    run_t0 = now_s();
    if (n_sched) apply_script(0); else apply_control();
    run();
    autopilot_flush();
    double el = now_s() - run_t0;
    uint32_t fc = fc_addr ? rd32(mem + fc_addr) : 0;
    if (guest_out && !strcmp(guest_out, "none")) {
        // discard guest console output
    } else if (guest_out) {
        FILE *f = fopen(guest_out, "wb");
        if (f) { fwrite(outbuf, 1, outlen, f); fclose(f); }
    } else {
        fwrite(outbuf, 1, outlen, stdout);
        fflush(stdout);
    }
    if (store_vals)
        fprintf(stderr, "fastemu-lf: consumed %zu of %zu store values\n",
                store_pos, n_store_vals);
    if (prof_fp) { fclose(prof_fp); prof_fp = NULL; }
    fprintf(stderr,
            "fastemu-lf: steps=%llu frames=%u elapsed=%.2fs (%.1f M instr/s, %.2f fps) "
            "halted=%d exit=%d fault=%s\n",
            (unsigned long long)steps, fc, el,
            steps / el / 1e6, fc / el, halted, exit_code,
            fault_cause >= 0 ? FAULT_NAMES[fault_cause] : "-");
    if (fault_cause >= 0)
        fprintf(stderr, "fastemu-lf: fault %s at pc=%#x\n",
                FAULT_NAMES[fault_cause], fault_pc);
    return fault_cause >= 0 ? 2 : 0;
}
