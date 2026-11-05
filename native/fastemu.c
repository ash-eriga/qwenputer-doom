// fastemu.c — a fast RV32IM + Zba interpreter for the Qwenputer Doom port.
//
// The reference machine (chipformer/rv/ref.py Machine) is the semantics
// oracle, but it builds the circuit's token trace for every instruction:
// ~490k instructions/s in CPython on the Mac. This interpreter executes
// the same semantics directly — ROM pre-decoded once into a compact
// struct array, computed-goto dispatch, flat byte memory — and produces
// no trace. It exists to run the Doom port (and shakeout programs) at
// full speed on the host; the in-weights circuit remains the real target
// and keeps the slow, honest trace.
//
// Semantics intentionally mirror ref.py Machine:
//   - memory map: RAM 2 MB @ 0x0, DROM 6 MB @ 0x200000, ROM 256 KB @
//     0x800000 (contiguous, so one flat 0x840000-byte array serves all
//     three; loads may read RAM/DROM only, stores may write RAM only)
//   - misaligned load/store faults, load/store out-of-range faults,
//     bad-jump faults (targets must be 4-aligned inside ROM)
//   - ecall a7=1 putchar(a0), a7=93 exit(a0); other ecalls fault
//   - DIV/REM edge cases (divide by zero, INT_MIN / -1) match ref.alu
// Not supported (absent from the Doom binary — verified by a static
// decode of all 58,432 ROM words): CSR instructions and the custom
// digit-file ops (BMAC/BMSB/BLD/BGET/BSCAN/BPRD). They decode as
// ILLEGAL here.
//
// The frame hook: qwen_frame_count is a word in RAM written with SW by
// DG_DrawFrame the moment a frame is complete, so the SW handler watches
// that one address — frame dumps, the live PNG and exact-stop hashes all
// happen at the precise instruction where the frame committed.
//
// Build:  cc -O3 -o fastemu fastemu.c -lz
// Usage:  fastemu doom-big.elf [--frames N] [--steps N] [--wad doom1.wad]
//                [--dump 10,100,120 --dump-dir DIR] [--live latest.png]
//                [--hash-at N] [--fb 0xCA3A8] [--fc 0xD9DB8] [--res 320x200]
//                [--guest-out FILE|none] [--pace FPS]
//                [--control FILE] [--keys 0xD9DB4] [--demo 0xADDR]
//
// The control file (8 bytes, mmap-shared with whatever serves the live
// page) hands the game to a human: bytes 0..3 = the qwen_key_state mask
// (bit per key: up/down/left/right/fire/use/enter/escape), byte 4 = mode
// (0 = demo drives, 1 = manual: the demo flag cell is cleared and the
// mask is poked into the key-state cell, both at frame boundaries).

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
#include <zlib.h>

#define RAM_SIZE  0x200000u
#define DROM_BASE 0x200000u
#define DROM_SIZE 0x600000u
#define ROM_BASE  0x800000u
#define ROM_SIZE  0x40000u
#define MEM_SIZE  (ROM_BASE + ROM_SIZE)

static const char *FAULT_NAMES[8] = {"illegal", "ebreak", "misaligned load",
    "misaligned store", "load out of range", "store out of range",
    "bad jump", "ecall"};

static uint8_t *mem;                 // flat MEM_SIZE bytes (calloc: bss = 0)
static uint32_t regs[32];
static uint32_t entry_pc, pc;
static uint64_t steps;
static int halted, exit_code, fault_cause = -1;
static uint32_t fault_pc;

// guest console output (ecall putchar), buffered and written at the end
static uint8_t *outbuf;
static size_t outlen, outcap;

static void out_push(uint8_t c) {
    if (outlen == outcap) {
        outcap = outcap ? 2 * outcap : 65536;
        outbuf = realloc(outbuf, outcap);
        if (!outbuf) { perror("realloc"); exit(1); }
    }
    outbuf[outlen++] = c;
}

// ---------------------------------------------------------------- ELF load

static uint32_t rd32(const uint8_t *p) {
    return p[0] | p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

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
        fprintf(stderr, "fastemu: not an ELF32 file\n");
        return -1;
    }
    entry_pc = rd32(b + 0x18);
    uint32_t phoff = rd32(b + 0x1C);
    int phentsize = b[0x2A] | b[0x2B] << 8, phnum = b[0x2C] | b[0x2D] << 8;
    for (int i = 0; i < phnum; i++) {
        const uint8_t *ph = b + phoff + i * phentsize;
        uint32_t type = rd32(ph), off = rd32(ph + 4), va = rd32(ph + 8);
        uint32_t filesz = rd32(ph + 16), memsz = rd32(ph + 20);
        if (type != 1 || !memsz) continue;           // PT_LOAD only
        if (va + memsz > MEM_SIZE) {
            fprintf(stderr, "fastemu: segment at %#x (+%#x) outside the map\n",
                    va, memsz);
            return -1;
        }
        memcpy(mem + va, b + off, filesz);           // bss tail stays zero
    }
    free(b);
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
static Ins code[ROM_SIZE / 4];

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
    for (uint32_t wi = 0; wi < ROM_SIZE / 4; wi++) {
        uint32_t w = rd32(mem + ROM_BASE + 4 * wi);
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
            if (w == 0x00000073u) in.op = O_ECALL;
            else if (w == 0x00100073u) in.op = O_EBREAK;
            break;                       // CSR forms: none in this binary
        case 0x0F: in.op = O_FENCE; break;
        default: break;                  // includes the digit-file opcodes
        }
        code[wi] = in;
    }
}

// ---------------------------------------------------------------- frames out

static uint8_t palette[768];
static int have_palette;
static uint32_t fb_addr = 0xCA3A8, fc_addr = 0xD9DB8;
static int fb_is_ptr;   /* --fbp: fb_addr holds a pointer to the framebuffer, not the buffer */
static uint32_t key_addr = 0xD9DB4, demo_addr;   // demo_addr 0 = not poked
static volatile uint8_t *ctrl;                  // mmap'd control file, 8 B
static int resx = 320, resy = 200;

static void apply_control(void) {
    if (!ctrl) return;
    uint32_t mask;
    memcpy(&mask, (const void *)ctrl, 4);
    uint8_t mode = ctrl[4];
    uint32_t v;
    if (demo_addr) {
        v = mode == 0 ? 1 : 0;
        memcpy(mem + demo_addr, &v, 4);
    }
    v = mode == 1 ? mask : 0;
    memcpy(mem + key_addr, &v, 4);
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

static void put32be(uint8_t *p, uint32_t v) {
    p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
}

static int write_png(const char *path) {
    int w = resx, h = resy;
    /* --fb may name a pointer variable (the engine allocates its screen at runtime): follow
       it, so a build whose framebuffer moves still dumps the frame it drew. */
    uint32_t base = fb_addr;
    if (fb_is_ptr) {
        base = rd32(mem + fb_addr);
        if (!base || base + (uint32_t)(w * h) > MEM_SIZE) return -1;
    }
    const uint8_t *fb = mem + base;
    uint8_t *raw = malloc(h * (1 + 3 * w));
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
    uLongf bound = compressBound(h * (1 + 3 * w));
    uint8_t *comp = malloc(bound);
    if (compress2(comp, &bound, raw, h * (1 + 3 * w), 1) != Z_OK) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { free(raw); free(comp); return -1; }
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    uint8_t lenb[4];
    uint8_t ihdr[17];                    // "IHDR" + 13 data bytes
    memcpy(ihdr, "IHDR", 4);
    put32be(ihdr + 4, w);
    put32be(ihdr + 8, h);
    ihdr[12] = 8; ihdr[13] = 2; ihdr[14] = 0; ihdr[15] = 0; ihdr[16] = 0;
    uint32_t crc = crc32(crc32(0, NULL, 0), ihdr, 17);
    put32be(lenb, 13);
    fwrite(lenb, 1, 4, f);
    fwrite(ihdr, 1, 17, f);
    put32be(lenb, crc);
    fwrite(lenb, 1, 4, f);
    // IDAT
    uint8_t *idat = malloc(4 + bound);
    memcpy(idat, "IDAT", 4);
    memcpy(idat + 4, comp, bound);
    crc = crc32(crc32(0, NULL, 0), idat, 4 + bound);
    put32be(lenb, bound);
    fwrite(lenb, 1, 4, f);
    fwrite(idat, 1, 4 + bound, f);
    put32be(lenb, crc);
    fwrite(lenb, 1, 4, f);
    // IEND
    put32be(lenb, 0);
    fwrite(lenb, 1, 4, f);
    fwrite("IEND", 1, 4, f);
    put32be(lenb, crc32(crc32(0, NULL, 0), (const uint8_t *)"IEND", 4));
    fwrite(lenb, 1, 4, f);
    fclose(f);
    free(raw); free(comp); free(idat);
    return 0;
}

static int write_ppm(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t base = fb_addr;
    if (fb_is_ptr) {
        base = rd32(mem + fb_addr);
        if (!base || base + (uint32_t)(resx * resy) > MEM_SIZE) { fclose(f); return -1; }
    }
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
    for (uint32_t i = 0; i < RAM_SIZE; i++) { h ^= mem[i]; h *= 1099511628211ULL; }
    for (int i = 0; i < 32; i++) {
        uint32_t v = regs[i];
        for (int k = 0; k < 4; k++) { h ^= v >> 8 * k & 0xFF; h *= 1099511628211ULL; }
    }
    for (int k = 0; k < 4; k++) { h ^= pc >> 8 * k & 0xFF; h *= 1099511628211ULL; }
    return h;
}

// ---------------------------------------------------------------- run config

static uint32_t target_frames, hash_at;
static uint64_t max_steps;
static double pace_fps;                  // >0: hold this wall-clock frame rate
static uint32_t dump_frames[64];
static int n_dumps;
static const char *dump_dir, *live_path;
static double run_t0, last_live_write = -1;
static uint32_t last_fc;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Called from the SW handler at the exact instruction where
// qwen_frame_count changed. Returns 1 to stop the run.
static int frame_hook(uint32_t fc) {
    apply_control();
    for (int i = 0; i < n_dumps; i++) {
        if (dump_frames[i] == fc && dump_dir && have_palette) {
            char p[1024];
            snprintf(p, sizeof p, "%s/frame-%04u.ppm", dump_dir, fc);
            write_ppm(p);
            fprintf(stderr, "fastemu: dumped frame %u\n", fc);
        }
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

static int in_rom(uint32_t t) {
    return !(t & 3) && t >= ROM_BASE && t < ROM_BASE + ROM_SIZE;
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
    regs[2] = RAM_SIZE;                  // SP_INIT: top of RAM

#define DISPATCH() do { ins = &code[(pc - ROM_BASE) >> 2]; goto *table[ins->op]; } while (0)
#define NEXT() do { steps++; if (--chunk == 0) { chunk = 8192; \
    if (max_steps && steps >= max_steps) goto finish; } DISPATCH(); } while (0)
#define FAULT(c) do { fault_cause = (c); fault_pc = pc; halted = 1; goto finish; } while (0)
#define WR(v) do { if (ins->rd) regs[ins->rd] = (uint32_t)(v); } while (0)

    DISPATCH();

L_LUI:   WR(ins->imm); pc += 4; NEXT();
L_AUIPC: WR(pc + (uint32_t)ins->imm); pc += 4; NEXT();
L_JAL: {
        uint32_t t = pc + (uint32_t)ins->imm;
        if (!in_rom(t)) FAULT(6);
        WR(pc + 4); pc = t; NEXT();
    }
L_JALR: {
        uint32_t t = (regs[ins->rs1] + (uint32_t)ins->imm) & ~1u;
        if (!in_rom(t)) FAULT(6);
        WR(pc + 4); pc = t; NEXT();
    }
L_BEQ:  if (regs[ins->rs1] == regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BNE:  if (regs[ins->rs1] != regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BLT:  if ((int32_t)regs[ins->rs1] < (int32_t)regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BGE:  if ((int32_t)regs[ins->rs1] >= (int32_t)regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BLTU: if (regs[ins->rs1] < regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();
L_BGEU: if (regs[ins->rs1] >= regs[ins->rs2]) { uint32_t t = pc + (uint32_t)ins->imm; if (!in_rom(t)) FAULT(6); pc = t; } else pc += 4; NEXT();

#define LOAD_OK(ea, n) (((ea) < RAM_SIZE && (ea) + (n) <= RAM_SIZE) || \
    ((ea) >= DROM_BASE && (ea) < DROM_BASE + DROM_SIZE && (ea) + (n) <= DROM_BASE + DROM_SIZE))
#define EA() (regs[ins->rs1] + (uint32_t)ins->imm)
L_LB: {
        uint32_t ea = EA();
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
        if (!(ea < RAM_SIZE)) FAULT(5);
        mem[ea] = regs[ins->rs2] & 0xFF; pc += 4; NEXT();
    }
L_SH: {
        uint32_t ea = EA();
        if (ea & 1) FAULT(3);
        if (!(ea < RAM_SIZE && ea + 2 <= RAM_SIZE)) FAULT(5);
        uint16_t v = regs[ins->rs2] & 0xFFFF; memcpy(mem + ea, &v, 2); pc += 4; NEXT();
    }
L_SW: {
        uint32_t ea = EA();
        if (ea & 3) FAULT(3);
        if (!(ea < RAM_SIZE && ea + 4 <= RAM_SIZE)) FAULT(5);
        uint32_t v = regs[ins->rs2];
        memcpy(mem + ea, &v, 4);
        pc += 4;
        steps++;
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
        if (a7 == 1) {
            uint32_t v = regs[10] & 0x1FF;
            if (v < 256) out_push((uint8_t)v);
            pc += 4; NEXT();
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
    const char *control_path = NULL;
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
        else if (!strcmp(a, "--fc") && i + 1 < argc) fc_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--keys") && i + 1 < argc) key_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--demo") && i + 1 < argc) demo_addr = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--control") && i + 1 < argc) control_path = argv[++i];
        else if (!strcmp(a, "--dump") && i + 1 < argc) {
            char *copy = strdup(argv[++i]), *tok = strtok(copy, ",");
            while (tok && n_dumps < 64) {
                dump_frames[n_dumps++] = strtoul(tok, NULL, 0);
                tok = strtok(NULL, ",");
            }
            free(copy);
        } else if (!strcmp(a, "--res") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &resx, &resy) != 2) {
                fprintf(stderr, "fastemu: bad --res\n");
                return 1;
            }
        } else if (a[0] != '-') elf_path = a;
        else { fprintf(stderr, "fastemu: unknown option %s\n", a); return 1; }
    }
    if (!elf_path) {
        fprintf(stderr, "usage: fastemu <elf> [options]\n");
        return 1;
    }
    if ((n_dumps || live_path) && !wad_path) {
        fprintf(stderr, "fastemu: --dump/--live need --wad for the palette\n");
        return 1;
    }
    mem = calloc(1, MEM_SIZE);
    if (!mem) { perror("calloc"); return 1; }
    if (load_elf(elf_path)) return 1;
    if (wad_path && load_palette(wad_path)) {
        fprintf(stderr, "fastemu: PLAYPAL not found in %s\n", wad_path);
        return 1;
    }
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
    apply_control();
    run();
    double el = now_s() - run_t0;
    uint32_t fc = rd32(mem + fc_addr);
    if (guest_out && !strcmp(guest_out, "none")) {
        // discard guest console output
    } else if (guest_out) {
        FILE *f = fopen(guest_out, "wb");
        if (f) { fwrite(outbuf, 1, outlen, f); fclose(f); }
    } else {
        fwrite(outbuf, 1, outlen, stdout);
        fflush(stdout);
    }
    fprintf(stderr,
            "fastemu: steps=%llu frames=%u elapsed=%.2fs (%.1f M instr/s, %.2f fps) "
            "halted=%d exit=%d fault=%s\n",
            (unsigned long long)steps, fc, el,
            steps / el / 1e6, fc / el, halted, exit_code,
            fault_cause >= 0 ? FAULT_NAMES[fault_cause] : "-");
    if (fault_cause >= 0)
        fprintf(stderr, "fastemu: fault %s at pc=%#x\n",
                FAULT_NAMES[fault_cause], fault_pc);
    return fault_cause >= 0 ? 2 : 0;
}
