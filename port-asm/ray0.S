# ray0: a token-cost probe for a Doom-like renderer written directly for the LX machine.
#
# The hot loop is shaped for the machine's token cost, not for a normal CPU:
#   * no division in the frame loop (tables/code generation instead)
#   * aligned SW only in the framebuffer hot path
#   * X/Y ray state packed in one register; one ADD advances both axes
#   * map256 lookup indexes the packed position directly
#   * column drawing is a generated straight-line kernel per distance: 16 SW, no loop
#
# Keys match fastemu-lf: bit 0 up, 1 down, 2 left, 3 right.

    .equ W, 96
    .equ H, 60
    .equ STRIDE, 64
    .equ GROUPS, 16              # 16 groups * 4 probes = 64 half-tile steps

#ifndef VISUAL
#define VISUAL 0
#endif
#ifndef RAY_SHIFT
#define RAY_SHIFT 4
#endif

.macro PROBE hitlabel
    add s5, s5, s6              # packed x/y += packed dx/dy
    srli t0, s5, 4              # xtile in low bits, ytile in bits 8+
    and t0, t0, a1              # keep xtile + ytile*256
    add t0, t0, a7              # map256 base (the costly high-address add)
    lbu s10, 0(t0)
    bne s10, zero, \hitlabel
.endm

    .section .text.init
    .globl _start
_start:
    .option push
    .option norelax
    la sp, __stack_top
    la gp, sin_tab
    la tp, cos_tab
    la s11, map
    la a5, fb_cols
    la a6, camoff_tab
    la a7, map256
    la a0, colkern_base
    .option pop

    li a1, 0x0f0f               # packed tile-index mask
    # Doom palette indexes, replicated to four pixels per word:
    # ceiling = 64 (tan brown), wall = 144/128 (brown/tan), floor = 96 (gray)
    li a2, 0x40404040
    li a3, 0x90909090
    li a4, 0x60606060

    # Player at tile (2.5, 2.5), facing east. Positions are 4.4 fixed point.
    li s0, 40                   # px (4.4 fixed: tile 2.5)
    li s1, 40                   # py (4.4 fixed)
    li s2, 0                    # angle, 0..255 (one turn)

main_loop:
    # ------------------------------------------------------------ input/movement
    # Default is a baked autopilot: the free-running model needs no host input after GO.
    # qwen_demo_enabled=0 switches to qwen_key_state, which fastemu uses for scripts/live play.
    la t0, qwen_demo_enabled
    lw t0, 0(t0)
    beq t0, zero, use_host_keys

    la t0, qwen_frame_count
    lw t0, 0(t0)
    andi t0, t0, 63             # 64-frame patrol cycle
    li t1, 16
    blt t0, t1, auto_forward
    li t1, 32
    blt t0, t1, auto_right
    li t1, 48
    blt t0, t1, auto_forward
    li t0, 5                    # forward + left
    j have_input
auto_right:
    li t0, 9                    # forward + right
    j have_input
auto_forward:
    li t0, 1                    # forward
    j have_input
use_host_keys:
    la t0, qwen_key_state
    lw t0, 0(t0)
have_input:

    andi t1, t0, 4              # left: turn -3/256 turn (~4.2 degrees)
    beq t1, zero, 1f
    addi s2, s2, -3
    andi s2, s2, 255
1:
    andi t1, t0, 8              # right: turn +3/256 turn
    beq t1, zero, 2f
    addi s2, s2, 3
    andi s2, s2, 255
2:
    # Movement vector for this frame: cos/sin in 8.8, shifted to 1/8 tile per frame.
    slli t1, s2, 1
    add t2, tp, t1
    lh t2, 0(t2)
    srai t3, t2, 6              # mx in 4.4 units (1/4 tile/frame max)
    add t2, gp, t1
    lh t2, 0(t2)
    srai t4, t2, 6              # my in 4.4 units

    andi t1, t0, 1              # up: forward
    beq t1, zero, 3f
    add t5, s0, t3
    add t6, s1, t4
    j 4f
3:
    andi t1, t0, 2              # down: backward
    beq t1, zero, 5f
    sub t5, s0, t3
    sub t6, s1, t4
4:
    # Axis-separated collision against the byte map. x first:
    srli t1, t5, 4
    srli t2, s1, 4
    slli t2, t2, 4
    add t1, t1, t2
    add t1, t1, s11
    lbu t2, 0(t1)
    bne t2, zero, 41f
    mv s0, t5
41:
    srli t1, s0, 4
    srli t2, t6, 4
    slli t2, t2, 4
    add t1, t1, t2
    add t1, t1, s11
    lbu t2, 0(t1)
    bne t2, zero, 5f
    mv s1, t6
5:

    # ------------------------------------------------------------ render one frame
    li s3, 0                    # column x
column_loop:
    # ray angle = player angle + camera offset[x]
    add t0, a6, s3
    lb t0, 0(t0)
    add t0, t0, s2
    andi t0, t0, 255
    slli t0, t0, 1

    add t1, tp, t0
    lh t1, 0(t1)
    srai s7, t1, RAY_SHIFT      # ray step x in 4.4 units
    add t1, gp, t0
    lh t1, 0(t1)
    srai s8, t1, RAY_SHIFT      # ray step y

    # Pack position and step as x | y<<8 in 4.4 fixed point. Coordinates stay below
    # 256 units, so low-byte carries cannot leak into the y byte inside the map.
    slli s5, s1, 8
    or s5, s5, s0
    slli t0, s7, 24
    srli t0, t0, 24             # zero-extend signed dx into the low byte
    slli t1, s8, 8
    or s6, t0, t1

    li s9, GROUPS
march_loop:
    PROBE hit0
    PROBE hit1
    PROBE hit2
    PROBE hit3
    addi s9, s9, -1
    bne s9, zero, march_loop
    li t0, 64                   # miss: maximum-distance case
    j have_steps

# steps = (16 - groups_left) * 4 + probe_index + 1 = 65 + probe_index - 4*groups_left
hit0:
    slli t0, s9, 2
    li t1, 65
    sub t0, t1, t0
    j have_steps
hit1:
    slli t0, s9, 2
    li t1, 66
    sub t0, t1, t0
    j have_steps
hit2:
    slli t0, s9, 2
    li t1, 67
    sub t0, t1, t0
    j have_steps
hit3:
    slli t0, s9, 2
    li t1, 68
    sub t0, t1, t0

have_steps:
    # t0 is now the distance case, 1..64. Column kernels encode ceiling/wall/floor
    # for that case, so there is no span arithmetic and no drawing loop here.
    slli s4, s3, 6              # column base: x * 64
    add s4, s4, a5

    # Cheap texture/shading probe: wall tint alternates by cell type and column parity.
    li a3, 0x90909090
    andi t1, s3, 1
    or t1, t1, s10
    andi t1, t1, 2
    beq t1, zero, 6f
    li a3, 0x80808080
6:
    # case stride is 80 bytes: case*80 = case*64 + case*16
    slli t1, t0, 4
    slli t2, t0, 6
    add t1, t1, t2
    add t1, t1, a0
    jr t1

    .globl colkern_done
colkern_done:
    addi s3, s3, 1
    li t0, W
    blt s3, t0, column_loop

#if VISUAL
    # Debug/visual builds only: transpose the column-major frame into the linear buffer
    # fastemu-lf knows how to dump. This loop is deliberately excluded from cost builds.
    la t0, fb_cols
    la t1, fb_linear
    li t2, 0                    # x
vis_x_loop:
    li t3, 0                    # y
vis_y_loop:
    slli t4, t2, 6
    add t4, t4, t3
    add t4, t4, t0
    lbu t5, 0(t4)
    li t4, W
    mul t4, t3, t4
    add t4, t4, t2
    add t4, t4, t1
    sb t5, 0(t4)
    addi t3, t3, 1
    li t4, H
    blt t3, t4, vis_y_loop
    addi t2, t2, 1
    li t4, W
    blt t2, t4, vis_x_loop
#endif

    # Commit the frame. fastemu-lf watches this exact store.
    la t0, qwen_frame_count
    lw t1, 0(t0)
    addi t1, t1, 1
    sw t1, 0(t0)

    j main_loop

    .section .bss
    .align 4
    .globl fb_cols
fb_cols:
    .skip W * STRIDE

    .globl fb_linear
fb_linear:
    .skip W * H

    .globl qwen_frame_count
qwen_frame_count:
    .word 0
    .globl qwen_key_state
qwen_key_state:
    .word 0

    .section .data
    .align 2
    .globl qwen_demo_enabled
qwen_demo_enabled:
    .word 1

    .globl DG_ScreenBuffer
DG_ScreenBuffer:
#if VISUAL
    .word fb_linear
#else
    .word fb_cols
#endif
