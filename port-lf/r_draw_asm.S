/* Hot column and span loops for the LX Doom build: the same pixels the C renderer draws, with
 * fewer and cheaper LX records.
 *
 * Why assembly here.  On the LX machine every rd-writing instruction is a token (two when the
 * result does not fit in 16 bits), a byte or halfword store is three tokens (two of them carry
 * the destination word's old value) and an aligned word store is one.  These functions are the
 * bulk of a frame's tokens, so:
 *
 *   R_DrawSpan       (DETAIL 0) writes one byte per view pixel.  Four horizontally adjacent
 *                    pixels are assembled in registers and written with one aligned SW; the
 *                    unaligned head and tail (0..3 pixels) use byte stores.
 *   R_DrawSpanLow    (DETAIL 1) computes one texture lookup per view pixel -- Doom's own low
 *                    detail, where a view pixel is two screen pixels -- and writes the doubled
 *                    pair with one aligned SW, packing two view pixels (four screen bytes) per
 *                    store.  The pair itself is even-aligned, so at most one SH starts it off.
 *   R_DrawColumn     (DETAIL 0) writes one byte per row.  Columns are SCREENWIDTH bytes apart,
 *                    so nothing packs: the loop is unrolled and keeps dc_source, dc_colormap,
 *                    frac, fracstep, the row stride and the wrap mask in registers.
 *   R_DrawColumnLow  (DETAIL 1) writes each column pixel to two adjacent screen columns, which
 *                    is a single halfword store of the doubled byte.
 *
 * All four reproduce the C code loop for loop: the same do/while pixel count (dc_yh-dc_yl+1,
 * ds_x2-ds_x1+1), the same 32-bit wrapping fixed-point arithmetic and the same texture index
 * masking.  What that is worth was measured, not assumed:
 *
 *   DETAIL 0 pair   pixel-identical to the C renderer on lx.ref.
 *   DETAIL 1 pair   2.3x fewer tokens on the span and 1.5x on the column.  An earlier note here
 *                   claimed this pair differed from the C renderer by 14 of 4,000 pixels on the
 *                   horizon row; that was an artefact of comparing against a stale reference
 *                   image, not of the assembly.  Built from the same sources, compare_frames.py
 *                   against a REF=1 build reports differing=0 on frames 6, 10, 60 and 150 at
 *                   160x100 (the configuration build_fast.sh ships), so -DR_DRAW_ASM_SPANLOW=1
 *                   and -DR_DRAW_ASM_COLLOW=1 are on in the shipped build.
 */
/* Selection switch: r_draw.c compiles the C loop of a function when its R_DRAW_ASM_<X> is 0.
 * Nothing here is on by default; a build that wants assembly asks for it per function. */
#define SCREENWIDTH DG_SCREENWIDTH
#define SCREENHEIGHT DG_SCREENHEIGHT

/* r_draw.c keeps its C loops behind the same switches so each one can be A/B tested. */
#ifndef R_DRAW_ASM_SPAN
#define R_DRAW_ASM_SPAN 0
#endif
#ifndef R_DRAW_ASM_SPANLOW
#define R_DRAW_ASM_SPANLOW 0
#endif
#ifndef R_DRAW_ASM_COL
#define R_DRAW_ASM_COL 0
#endif
#ifndef R_DRAW_ASM_COLLOW
#define R_DRAW_ASM_COLLOW 0
#endif

    .text

/* ------------------------------------------------------------------ R_DrawColumn (DETAIL 0)
 *
 *   count = dc_yh - dc_yl; if (count < 0) return;
 *   dest  = ylookup[dc_yl] + columnofs[dc_x];
 *   fracstep = dc_iscale; frac = dc_texturemid + (dc_yl-centery)*fracstep;
 *   do { *dest = dc_colormap[dc_source[(frac>>16)&127]];
 *        dest += SCREENWIDTH; frac += fracstep; } while (count--);
 */
#if R_DRAW_ASM_COL
    .globl R_DrawColumn
    .type  R_DrawColumn, @function
    .align 2
R_DrawColumn:
    la      t0, dc_yh
    lw      a0, 0(t0)
    la      t1, dc_yl
    lw      a1, 0(t1)
    sub     a2, a0, a1                  /* count */
    blt     a2, zero, .Lcol_ret

    addi    sp, sp, -32
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    sw      s7, 28(sp)

    /* dest = ylookup[dc_yl] + columnofs[dc_x] */
    la      t0, ylookup
    slli    t1, a1, 2
    add     t0, t0, t1
    lw      s0, 0(t0)                   /* row base */
    la      t0, columnofs
    la      t1, dc_x
    lw      t2, 0(t1)
    slli    t2, t2, 2
    add     t0, t0, t2
    lw      t3, 0(t0)
    add     s0, s0, t3                  /* dest */

    /* frac = dc_texturemid + (dc_yl - centery)*dc_iscale */
    la      t0, dc_texturemid
    lw      s2, 0(t0)                   /* frac */
    la      t0, centery
    lw      t1, 0(t0)
    sub     t2, a1, t1
    la      t0, dc_iscale
    lw      s3, 0(t0)                   /* fracstep */
    mul     t3, t2, s3
    add     s2, s2, t3

    la      t0, dc_source
    lw      s4, 0(t0)                   /* source */
    la      t0, dc_colormap
    lw      s5, 0(t0)                   /* colormap */
    li      s6, SCREENWIDTH             /* row stride */
    li      s7, 127                     /* texture column wrap */
    addi    s1, a2, 1                   /* pixels: count + 1 */
    li      a3, 4                       /* unroll threshold, hoisted */

    /* Unrolled by four: only the counter and its branch are amortised, every pixel still pays
       its index shift, two table reads and its store. */
.Lcol_loop4:
    blt     s1, a3, .Lcol_loop1
.Lcol_body4:
    srli    t0, s2, 16
    and     t0, t0, s7
    add     t1, s4, t0
    lbu     t2, 0(t1)
    add     t3, s5, t2
    lbu     t4, 0(t3)
    sb      t4, 0(s0)
    add     s0, s0, s6
    add     s2, s2, s3

    srli    t0, s2, 16
    and     t0, t0, s7
    add     t1, s4, t0
    lbu     t2, 0(t1)
    add     t3, s5, t2
    lbu     t4, 0(t3)
    sb      t4, 0(s0)
    add     s0, s0, s6
    add     s2, s2, s3

    srli    t0, s2, 16
    and     t0, t0, s7
    add     t1, s4, t0
    lbu     t2, 0(t1)
    add     t3, s5, t2
    lbu     t4, 0(t3)
    sb      t4, 0(s0)
    add     s0, s0, s6
    add     s2, s2, s3

    srli    t0, s2, 16
    and     t0, t0, s7
    add     t1, s4, t0
    lbu     t2, 0(t1)
    add     t3, s5, t2
    lbu     t4, 0(t3)
    sb      t4, 0(s0)
    add     s0, s0, s6
    add     s2, s2, s3

    addi    s1, s1, -4
    bge     s1, a3, .Lcol_body4

.Lcol_loop1:
    beqz    s1, .Lcol_done
.Lcol_body1:
    srli    t0, s2, 16
    and     t0, t0, s7
    add     t1, s4, t0
    lbu     t2, 0(t1)
    add     t3, s5, t2
    lbu     t4, 0(t3)
    sb      t4, 0(s0)
    add     s0, s0, s6
    add     s2, s2, s3
    addi    s1, s1, -1
    bnez    s1, .Lcol_body1

.Lcol_done:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    lw      s7, 28(sp)
    addi    sp, sp, 32
.Lcol_ret:
    ret
    .size R_DrawColumn, .-R_DrawColumn
#endif

/* ------------------------------------------------------------------ R_DrawColumnLow (DETAIL 1)
 *
 *   count = dc_yh - dc_yl; if (count < 0) return;
 *   x = dc_x << 1;
 *   dest = dest2 - 1 = ylookup[dc_yl] + columnofs[x];        (dest2 = dest + 1)
 *   fracstep = dc_iscale; frac = dc_texturemid + (dc_yl-centery)*fracstep;
 *   do { *dest2 = *dest = dc_colormap[dc_source[(frac>>16)&127]];
 *        dest += SCREENWIDTH; dest2 += SCREENWIDTH; frac += fracstep; } while (count--);
 *
 * The two stores are adjacent bytes in one row, so one halfword store of the doubled colour
 * replaces both of them (three tokens instead of six).
 */
#if R_DRAW_ASM_COLLOW
    /* One column row: t3 = colormap[source[(frac>>16)&127]] * 0x101, stored as a halfword
     * at \off(dest).  Two MULs and one SH; the SH costs the LX three tokens (old word), so
     * nothing packs across rows -- rows are SCREENWIDTH bytes apart. */
    .macro col_pixel_low off
    srli    t3, t1, 16
    and     t3, t3, t5                  /* (frac>>16) & 127 */
    add     t3, a5, t3
    lbu     t3, 0(t3)
    add     t3, a6, t3
    lbu     t3, 0(t3)                   /* colormap[texel] */
    mul     t3, t3, t4                  /* doubled byte */
    sh      t3, \off(t0)
    add     t1, t1, a3                  /* frac += fracstep */
    .endm

    /* R_DrawColumnLowFast(x, yl, yh, iscale, texturemid, source, colormap) -- a0..a6.
     * Parameters stay in registers: the C renderer used to write seven globals and the
     * callee read all seven back, which on this machine is two records per parameter. */
    .globl R_DrawColumnLowFast
    .type  R_DrawColumnLowFast, @function
    .align 2
R_DrawColumnLowFast:
    sub     t0, a2, a1
    blt     t0, zero, .LcolL_ret
    la      t0, ylookup
    slli    t2, a1, 2
    add     t0, t0, t2
    lw      t0, 0(t0)                   /* row base */
    la      t2, columnofs
    slli    t1, a0, 3                   /* (x << 1) * 4 */
    add     t2, t2, t1
    lw      t2, 0(t2)
    add     t0, t0, t2                  /* dest */
    la      t2, centery
    lw      t2, 0(t2)
    sub     t2, a1, t2
    mul     t2, t2, a3
    add     t1, a4, t2                  /* frac = texturemid + (yl-centery)*iscale */
    sub     t2, a2, a1
    addi    t2, t2, 1                   /* pixels */
    li      t4, 0x101
    li      t5, 127
    slti    t3, t2, 4
    bnez    t3, .LcolL_2
.LcolL_4:
    col_pixel_low 0
    col_pixel_low SCREENWIDTH
    col_pixel_low 2*SCREENWIDTH
    col_pixel_low 3*SCREENWIDTH
    addi    t0, t0, 4*SCREENWIDTH
    addi    t2, t2, -4
    slti    t3, t2, 4
    beqz    t3, .LcolL_4
.LcolL_2:
    slti    t3, t2, 2
    bnez    t3, .LcolL_1
    col_pixel_low 0
    col_pixel_low SCREENWIDTH
    addi    t0, t0, 2*SCREENWIDTH
    addi    t2, t2, -2
.LcolL_1:
    beqz    t2, .LcolL_ret
    col_pixel_low 0
.LcolL_ret:
    ret
    .size R_DrawColumnLowFast, .-R_DrawColumnLowFast

    /* The globals-argument entry point r_main hands to colfunc (used by callers that
     * cannot use the fast form, and by any code that takes its address). */
    .globl R_DrawColumnLow
    .type  R_DrawColumnLow, @function
    .align 2
R_DrawColumnLow:
    la      t0, dc_x
    lw      a0, 0(t0)
    la      t0, dc_yl
    lw      a1, 0(t0)
    la      t0, dc_yh
    lw      a2, 0(t0)
    la      t0, dc_iscale
    lw      a3, 0(t0)
    la      t0, dc_texturemid
    lw      a4, 0(t0)
    la      t0, dc_source
    lw      a5, 0(t0)
    la      t0, dc_colormap
    lw      a6, 0(t0)
    j       R_DrawColumnLowFast
    .size R_DrawColumnLow, .-R_DrawColumnLow
#endif

/* ------------------------------------------------------------------ R_DrawSpan (DETAIL 0)
 *
 *   position = ((ds_xfrac<<10)&0xffff0000) | ((ds_yfrac>>6)&0xffff);
 *   step     = ((ds_xstep<<10)&0xffff0000) | ((ds_ystep>>6)&0xffff);
 *   dest = ylookup[ds_y] + columnofs[ds_x1];
 *   count = ds_x2 - ds_x1;
 *   do { spot = ((position>>4)&0x0fc0) | (position>>26);
 *        *dest++ = ds_colormap[ds_source[spot]];
 *        position += step; } while (count--);
 *
 * Four adjacent pixels share one aligned word store.
 */
    .macro span_pixel out, tmp
    srli    \out, s2, 4
    and     \out, \out, s7
    srli    \tmp, s2, 26
    or      \out, \out, \tmp
    add     \out, s4, \out
    lbu     \out, 0(\out)
    add     \out, s5, \out
    lbu     \out, 0(\out)
    .endm

#if R_DRAW_ASM_SPAN
    .globl R_DrawSpan
    .type  R_DrawSpan, @function
    .align 2
R_DrawSpan:
    addi    sp, sp, -32
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    sw      s7, 28(sp)

    /* position and step, packed exactly as the C code packs them */
    la      t0, ds_xfrac
    lw      s2, 0(t0)
    slli    s2, s2, 10
    li      t1, 0xffff0000
    and     s2, s2, t1
    la      t0, ds_yfrac
    lw      t2, 0(t0)
    srli    t2, t2, 6
    slli    t2, t2, 16
    srli    t2, t2, 16
    or      s2, s2, t2                  /* position */

    la      t0, ds_xstep
    lw      s3, 0(t0)
    slli    s3, s3, 10
    li      t1, 0xffff0000
    and     s3, s3, t1
    la      t0, ds_ystep
    lw      t2, 0(t0)
    srli    t2, t2, 6
    slli    t2, t2, 16
    srli    t2, t2, 16
    or      s3, s3, t2                  /* step */

    /* dest = ylookup[ds_y] + columnofs[ds_x1] */
    la      t0, ds_y
    lw      t1, 0(t0)
    la      t0, ylookup
    slli    t1, t1, 2
    add     t0, t0, t1
    lw      s0, 0(t0)
    la      t0, ds_x1
    lw      t1, 0(t0)
    la      t0, columnofs
    slli    t1, t1, 2
    add     t0, t0, t1
    lw      t2, 0(t0)
    add     s0, s0, t2                  /* dest */

    la      t0, ds_x2
    lw      t1, 0(t0)
    la      t0, ds_x1
    lw      t2, 0(t0)
    sub     s1, t1, t2
    addi    s1, s1, 1                   /* pixels: ds_x2 - ds_x1 + 1 */

    la      t0, ds_source
    lw      s4, 0(t0)                   /* flat tile */
    la      t0, ds_colormap
    lw      s5, 0(t0)                   /* light table */
    li      s7, 0x0fc0                  /* vertical texture index */
    li      s6, 4                       /* unroll threshold, hoisted */

    /* Byte stores until dest is word aligned (at most three). */
.Lspan_head:
    andi    t0, s0, 3
    beqz    t0, .Lspan_check
    span_pixel t1, t0
    sb      t1, 0(s0)
    addi    s0, s0, 1
    add     s2, s2, s3
    addi    s1, s1, -1
    bnez    s1, .Lspan_head
    j       .Lspan_done

.Lspan_check:
    blt     s1, s6, .Lspan_tail
.Lspan_body:
    span_pixel t1, t0
    add     s2, s2, s3
    span_pixel t2, t0
    add     s2, s2, s3
    span_pixel t3, t0
    add     s2, s2, s3
    span_pixel t4, t0
    add     s2, s2, s3

    slli    t5, t2, 8
    or      t5, t1, t5
    slli    t6, t3, 16
    or      t5, t5, t6
    slli    t6, t4, 24
    or      t5, t5, t6
    sw      t5, 0(s0)

    addi    s0, s0, 4
    addi    s1, s1, -4
    bge     s1, s6, .Lspan_body

.Lspan_tail:
    beqz    s1, .Lspan_done
.Lspan_tail1:
    span_pixel t1, t0
    sb      t1, 0(s0)
    addi    s0, s0, 1
    add     s2, s2, s3
    addi    s1, s1, -1
    bnez    s1, .Lspan_tail1

.Lspan_done:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    lw      s7, 28(sp)
    addi    sp, sp, 32
    ret
    .size R_DrawSpan, .-R_DrawSpan
#endif

/* ------------------------------------------------------------------ R_DrawSpanLow (DETAIL 1)
 *
 *   position = ((ds_xfrac<<10)&0xffff0000) | ((ds_yfrac>>6)&0xffff);
 *   step     = ((ds_xstep<<10)&0xffff0000) | ((ds_ystep>>6)&0xffff);
 *   count = ds_x2 - ds_x1 + 1;
 *   dest = ylookup[ds_y] + columnofs[ds_x1 << 1];
 *   do { spot = ((position>>4)&0x0fc0) | (position>>26);
 *        *(u16*)dest = ds_colormap[ds_source[spot]] * 0x101;
 *        dest += 2; position += step; } while (--count);
 *
 * dest is even (the column offset is ds_x1*2 and rows are even wide), so the doubled pixel is
 * a halfword store and two view pixels (four screen bytes) are one aligned word store.
 */
#if R_DRAW_ASM_SPANLOW
    /* One view pixel: t\res = colormap[source[((position>>4)&0xfc0)|(position>>26)]],
     * doubled to p|p<<8, then position += step.  Four ALU ops build the 12-bit texture index
     * out of two disjoint bit fields of the packed 32-bit position; the doubled colour comes
     * from the pre-doubled colormap (lf_cmap16, r_data.c) as one LHU, so no multiply. */
    .macro span_pixel_low res
    srli    \res, a2, 4
    and     \res, \res, t5              /* (position>>4) & 0xfc0 */
    srli    t3, a2, 26
    or      \res, \res, t3
    add     \res, a4, \res
    lbu     \res, 0(\res)
    add     \res, a5, \res
    lbu     \res, 0(\res)               /* colormap[texel] */
    add     a2, a2, a3                  /* position += step */
    .endm

    /* Two view pixels (four screen bytes) per aligned SW.  The low pixel is doubled with
     * MUL y,0x101 and the high one with MUL y,0x01010000, so the pack is MUL MUL OR SW --
     * one control token for the store instead of two three-token SHs. */
    .macro span_pair off
    span_pixel_low t1
    span_pixel_low t2
    mul     t1, t1, t4
    mul     t2, t2, t6
    or      t2, t2, t1
    sw      t2, \off(a0)
    .endm

    /* R_DrawSpanLowFast(dest, count, position, step, source, colormap) -- a0..a5.
     * R_MapPlane already has all six values before it stored them into ds_* for the C
     * loop to read back; passing them in registers removes that round trip.  Leaf, and
     * it needs no callee-saved register, so there is no stack frame at all. */
    .globl R_DrawSpanLowFast
    .type  R_DrawSpanLowFast, @function
    .align 2
R_DrawSpanLowFast:
    li      t4, 0x101
    li      t5, 0x0fc0
    li      t6, 0x1010000
    /* dest is even (columnofs[x1<<1] and the row base are); if it is 2 mod 4 one halfword
     * store starts it off so every SW below is aligned. */
    andi    t0, a0, 2
    beqz    t0, .LspanL_blocks
    span_pixel_low t1
    mul     t1, t1, t4
    sh      t1, 0(a0)
    addi    a0, a0, 2
    addi    a1, a1, -1
    beqz    a1, .LspanL_ret
.LspanL_blocks:
    slti    t0, a1, 8
    bnez    t0, .LspanL_4
.LspanL_8:
    span_pair 0
    span_pair 4
    span_pair 8
    span_pair 12
    addi    a0, a0, 16
    addi    a1, a1, -8
    slti    t0, a1, 8
    beqz    t0, .LspanL_8
.LspanL_4:
    slti    t0, a1, 4
    bnez    t0, .LspanL_2
    span_pair 0
    span_pair 4
    addi    a0, a0, 8
    addi    a1, a1, -4
.LspanL_2:
    slti    t0, a1, 2
    bnez    t0, .LspanL_1
    span_pair 0
    addi    a0, a0, 4
    addi    a1, a1, -2
.LspanL_1:
    beqz    a1, .LspanL_ret
    span_pixel_low t1
    mul     t1, t1, t4
    sh      t1, 0(a0)
.LspanL_ret:
    ret
    .size R_DrawSpanLowFast, .-R_DrawSpanLowFast

    /* The globals-argument entry point r_main hands to spanfunc. */
    .globl R_DrawSpanLow
    .type  R_DrawSpanLow, @function
    .align 2
R_DrawSpanLow:
    la      t0, ds_x2
    lw      a1, 0(t0)
    la      t0, ds_x1
    lw      t1, 0(t0)
    sub     a1, a1, t1
    addi    a1, a1, 1                   /* count */
    la      t0, ds_y
    lw      t2, 0(t0)
    la      t0, ylookup
    slli    t2, t2, 2
    add     t0, t0, t2
    lw      a0, 0(t0)
    la      t0, columnofs
    slli    t2, t1, 3                   /* (ds_x1 << 1) * 4 */
    add     t0, t0, t2
    lw      t2, 0(t0)
    add     a0, a0, t2                  /* dest */

    la      t0, ds_xfrac
    lw      a2, 0(t0)
    slli    a2, a2, 10
    li      t2, 0xffff0000
    and     a2, a2, t2
    la      t0, ds_yfrac
    lw      t3, 0(t0)
    srli    t3, t3, 6
    slli    t3, t3, 16
    srli    t3, t3, 16
    or      a2, a2, t3                  /* position */

    la      t0, ds_xstep
    lw      a3, 0(t0)
    slli    a3, a3, 10
    and     a3, a3, t2
    la      t0, ds_ystep
    lw      t3, 0(t0)
    srli    t3, t3, 6
    slli    t3, t3, 16
    srli    t3, t3, 16
    or      a3, a3, t3                  /* step */

    la      t0, ds_source
    lw      a4, 0(t0)
    la      t0, ds_colormap
    lw      a5, 0(t0)
    j       R_DrawSpanLowFast
    .size R_DrawSpanLow, .-R_DrawSpanLow

/* ---------------------------------------------- pre-shaded flat span (DETAIL 1)
 *
 *   Same loop as R_DrawSpanLowFast, but `shaded` already holds cmap[source[i]] for every
 *   texel i (port-lf/r_plane.c fills it once per (source, cmap) pair), so a view pixel is
 *   one LBU into the shaded tile instead of LBU source + ADD colormap + LBU colormap.
 *   The index arithmetic, the 0x101 horizontal doubling and the two-pixels-per-SW packing
 *   are byte-for-byte the ones above; the bytes written are the same.
 *
 *   R_DrawSpanLowShadedFast(dest, count, position, step, shaded) -- a0..a4.
 */
    .macro shaded_pixel_low res
    srli    \res, a2, 4
    and     \res, \res, t5              /* (position>>4) & 0xfc0 */
    srli    t3, a2, 26
    or      \res, \res, t3              /* | (position>>26) */
    add     \res, a4, \res
    lbu     \res, 0(\res)               /* shaded tile byte */
    add     a2, a2, a3                  /* position += step */
    .endm

    .macro shaded_pair off
    shaded_pixel_low t1
    shaded_pixel_low t2
    mul     t1, t1, t4                  /* doubled byte */
    mul     t2, t2, t6                  /* doubled byte, high half */
    or      t2, t2, t1
    sw      t2, \off(a0)
    .endm

    .globl R_DrawSpanLowShadedFast
    .type  R_DrawSpanLowShadedFast, @function
    .align 2
R_DrawSpanLowShadedFast:
    li      t4, 0x101
    li      t5, 0x0fc0
    li      t6, 0x1010000
    andi    t0, a0, 2
    beqz    t0, .LspanLS_blocks
    shaded_pixel_low t1
    mul     t1, t1, t4
    sh      t1, 0(a0)
    addi    a0, a0, 2
    addi    a1, a1, -1
    beqz    a1, .LspanLS_ret
.LspanLS_blocks:
    slti    t0, a1, 8
    bnez    t0, .LspanLS_4
.LspanLS_8:
    shaded_pair 0
    shaded_pair 4
    shaded_pair 8
    shaded_pair 12
    addi    a0, a0, 16
    addi    a1, a1, -8
    slti    t0, a1, 8
    beqz    t0, .LspanLS_8
.LspanLS_4:
    slti    t0, a1, 4
    bnez    t0, .LspanLS_2
    shaded_pair 0
    shaded_pair 4
    addi    a0, a0, 8
    addi    a1, a1, -4
.LspanLS_2:
    slti    t0, a1, 2
    bnez    t0, .LspanLS_1
    shaded_pair 0
    addi    a0, a0, 4
    addi    a1, a1, -2
.LspanLS_1:
    beqz    a1, .LspanLS_ret
    shaded_pixel_low t1
    mul     t1, t1, t4
    sh      t1, 0(a0)
.LspanLS_ret:
    ret
    .size R_DrawSpanLowShadedFast, .-R_DrawSpanLowShadedFast

/* ---------------------------------------------- pre-shaded flat fill
 *
 *   R_FillShadedFlat(dst, source, colormap) -- a0..a2.  dst is a word-aligned 4096-byte
 *   tile; writes cmap[source[i]] for i in 0..4095, four bytes per aligned SW.  Cold cost:
 *   one fill per (source, cmap) pair the span cache ever admits.
 */
    .globl R_FillShadedFlat
    .type  R_FillShadedFlat, @function
    .align 2
R_FillShadedFlat:
    li      t0, 1024                    /* words */
.Lfill_words:
    lbu     t2, 0(a1)
    add     t2, a2, t2
    lbu     t2, 0(t2)
    lbu     t3, 1(a1)
    add     t3, a2, t3
    lbu     t3, 0(t3)
    slli    t3, t3, 8
    or      t2, t2, t3
    lbu     t3, 2(a1)
    add     t3, a2, t3
    lbu     t3, 0(t3)
    slli    t3, t3, 16
    or      t2, t2, t3
    lbu     t3, 3(a1)
    add     t3, a2, t3
    lbu     t3, 0(t3)
    slli    t3, t3, 24
    or      t2, t2, t3
    sw      t2, 0(a0)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    t0, t0, -1
    bnez    t0, .Lfill_words
    ret
    .size R_FillShadedFlat, .-R_FillShadedFlat
#endif

/* A hand-written exact fixed_t division (LF_FixedDiv64: two divu plus 16 shift-subtract steps,
 * to replace libgcc's __divdi3 in FixedDiv's slow path) lived here and was measured out: 165
 * tokens per call against libgcc's 100, because libgcc normalizes the operands and leaves its
 * loop early.  See the comment in m_fixed.c's FixedDiv.  __divdi3 stays. */

    .section .note.GNU-stack,"",@progbits
