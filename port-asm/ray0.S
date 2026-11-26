# ray0: a token-cost probe for a Doom-like renderer written directly for the LX machine.
#
# The hot loop is shaped for the machine's token cost, not for a normal CPU:
#   * no division in the frame loop (tables/code generation instead)
#   * aligned SW only in the framebuffer hot path
#   * ray state is one linear postab address; one ADD advances both axes
#   * solid window edges make that single-ADD update safe without bounds checks
#   * column drawing is a generated straight-line kernel per distance: 15 SW, no loop
#
# Keys match fastemu-lf: bit 0 up, 1 down, 2 left, 3 right.

    .equ W, 96
    .equ H, 60
    .equ STRIDE, 64
    .equ GROUPS, 16              # 16 groups * 4 probes = 64 ray steps

#ifndef VISUAL
#define VISUAL 0
#endif
#ifndef RAY_SHIFT
#define RAY_SHIFT 6             # 8.8 trig -> quarter-tile units; full step is one tile
#endif

.macro PROBE hitlabel
    add s5, s5, s6              # postab address += dxsub + dysub*64
    lbu s10, 0(s5)
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
    la a7, postab
    la a0, colkern_base + 4160   # one past the maximum-distance kernel case
    la a1, raydelta_tab
    .option pop

    # Real quantized E1M1 colors generated into render_words.  Keep all seven live in
    # registers; generated column kernels use them directly and never reload colors.
    la t0, render_words
    lw a2, 0(t0)                # ceiling
    lw a4, 4(t0)                # floor
    lw a3, 8(t0)                # texture cluster 0
    lw t3, 12(t0)               # texture cluster 1
    lw t4, 16(t0)               # texture cluster 2
    lw t5, 20(t0)               # texture cluster 3
    lw t6, 24(t0)               # texture cluster 4

    # Player at the center of the extracted E1M1 window. Positions are quarter-tile units.
    li s0, 34                   # px = 8.5 tiles * 4
    li s1, 34                   # py = 8.5 tiles * 4
    li s2, 64                   # Doom spawn angle 90deg, in 1/256-turn units

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
    # Movement vector for this frame: cos/sin in 8.8, shifted to quarter-tile units.
    slli t1, s2, 1
    add t2, tp, t1
    lh t2, 0(t2)
    srai s7, t2, 8              # mx in quarter-tile units (1/4 tile/frame max)
    add t2, gp, t1
    lh t2, 0(t2)
    srai s8, t2, 8              # my in quarter-tile units

    andi t1, t0, 1              # up: forward
    beq t1, zero, 3f
    add s9, s0, s7
    add s10, s1, s8
    j 4f
3:
    andi t1, t0, 2              # down: backward
    beq t1, zero, 5f
    sub s9, s0, s7
    sub s10, s1, s8
4:
    # Axis-separated collision against the compact byte map. x first:
    srli t1, s9, 2
    srli t2, s1, 2
    slli t2, t2, 4
    add t1, t1, t2
    add t1, t1, s11
    lbu t2, 0(t1)
    bne t2, zero, 41f
    mv s0, s9
41:
    srli t1, s0, 2
    srli t2, s10, 2
    slli t2, t2, 4
    add t1, t1, t2
    add t1, t1, s11
    lbu t2, 0(t1)
    bne t2, zero, 5f
    mv s1, s10
5:

    # ------------------------------------------------------------ render one frame
    slli s7, s2, 1              # player angle as a byte offset into raydelta_tab
    add s7, s7, a1              # biased table needs no per-column modulo/mask
    slli s8, s1, 6
    add s8, s8, s0
    add s8, s8, a7              # postab address of the player position
    mv s4, a5                   # framebuffer column pointer, advanced by 64 per column
    li s3, 0                    # column x
column_loop:
    # ray angle = player angle + camera offset[x], both pre-doubled byte offsets.
    add t0, a6, s3
    lb t0, 0(t0)
    add t0, t0, s7
    lh s6, 0(t0)                # combined dxsub + dysub*64
    mv s5, s8                   # player postab address

    li s9, 4096                 # 16 march groups, pre-scaled by 256 bytes
march_loop:
    PROBE hit0
    PROBE hit1
    PROBE hit2
    PROBE hit3
    addi s9, s9, -256
    bne s9, zero, march_loop
    addi t0, a0, -64
    jr t0                       # miss: maximum-distance case 64

# a0=colkern_base+4160 and s9=256*groups_left, so a0-s9 is exactly case hit0.
hit0:
    sub t0, a0, s9
    jr t0
hit1:
    addi t0, s9, -64
    sub t0, a0, t0
    jr t0
hit2:
    addi t0, s9, -128
    sub t0, a0, t0
    jr t0
hit3:
    addi t0, s9, -192
    sub t0, a0, t0
    jr t0

    .globl colkern_done
colkern_done:
    addi s4, s4, 64
    addi s3, s3, 1
    li t0, W
    blt s3, t0, column_loop

#if VISUAL
    # Debug/visual builds only: transpose the column-major frame into the linear buffer
    # fastemu-lf knows how to dump. This loop is deliberately excluded from cost builds.
    la t0, fb_cols
    la t1, fb_linear
    li s7, 0                    # x
vis_x_loop:
    li s8, 0                    # y
vis_y_loop:
    slli s9, s7, 6
    add s9, s9, s8
    add s9, s9, t0
    lbu s10, 0(s9)
    li s9, W
    mul s9, s8, s9
    add s9, s9, s7
    add s9, s9, t1
    sb s10, 0(s9)
    addi s8, s8, 1
    li s9, H
    blt s8, s9, vis_y_loop
    addi s7, s7, 1
    li s9, W
    blt s7, s9, vis_x_loop
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
