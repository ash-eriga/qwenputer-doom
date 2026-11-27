/* linuxformer (LX machine) layout for the pruned E1M1 build.
 *
 * Same machine model as doom-lf.ld -- one contiguous image at 0x80000000, code fetched from the
 * ROM copy below TEXT_END, everything read as data above it -- but every size here is a lever:
 * the circuit allocates one RAM key head per 2048 words of the image, so RAM_SIZE, and nothing
 * else, decides how many layers and heads the model has. The guest image therefore holds only
 * what booting straight into E1M1 and playing it needs: the pruned WAD (prune_wad.py), 192x120
 * with no second framebuffer (DG_SCREEN_IS_VIDEOBUFFER), a zone sized to the level, and the
 * smallest heap and stack that still run.
 *
 * Override at link time with --defsym, e.g.
 *     -Wl,--defsym,RAM_SIZE=0x180000 -Wl,--defsym,STACK_BYTES=0x4000
 *
 *   [0x80000000, +TEXT_END)              .text (code, fetched from the ROM copy)
 *   [0x80000000 + TEXT_END, +RAM_SIZE)   .rodata, small data, .data, .tdata/.tbss, .bss,
 *                                        heap, stack
 *
 * TEXT_END and RAM_SIZE must match lx.core.Config(text_end=..., ram_size=...).
 *
 * Small static objects (.srodata/.sdata/.sbss, everything -msmall-data-limit=8 puts there) are
 * one contiguous group with __global_pointer$ anchored 2048 bytes into it, which is what lets
 * the linker turn every auipc+addi+load of a global scalar into a single gp-relative access
 * (crt0.S loads gp, and build_fast.sh links with --relax).  The group must stay under 4096
 * bytes; it is currently ~2.6 KB.
 */
ENTRY(_start)

TEXT_END    = DEFINED(TEXT_END)    ? TEXT_END    : 0x40000;   /* circuit packs 256 KB of code */
RAM_SIZE    = DEFINED(RAM_SIZE)    ? RAM_SIZE    : 0x180000;  /* 1.5 MB image */
STACK_BYTES = DEFINED(STACK_BYTES) ? STACK_BYTES : 0x4000;    /* 16 KB of stack */
HEAP_BYTES  = DEFINED(HEAP_BYTES)  ? HEAP_BYTES  : 0x4000;    /* 16 KB for picolibc malloc */

MEMORY
{
  ROM (rx) : ORIGIN = 0x80000000,            LENGTH = TEXT_END
  RAM (rw) : ORIGIN = 0x80000000 + TEXT_END, LENGTH = RAM_SIZE - TEXT_END
}

SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM

  .rodata : {
    *(.rodata .rodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  } > RAM

  /* Small data: .srodata/.sdata/.sbss are grouped contiguously so the RISC-V linker can relax
   * every load/store of a small global into a single gp-relative instruction instead of
   * auipc+addi+load (each record is an LX token).  gp points 2048 bytes into the group, so the
   * whole group must stay under 4096 bytes; .sbss is NOBITS and the guest starts from a zeroed
   * image, so it needs no initialiser. */
  .sdata : {
    *(.srodata .srodata.*)
    *(.sdata .sdata.*)
  } > RAM
  .sbss : { *(.sbss .sbss.*) } > RAM
  PROVIDE(__global_pointer$ = ADDR(.sdata) + 0x800);

  .data : { *(.data .data.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM

  /* picolibc's __fallback_sbrk hands out [__heap_start, __heap_end); the stack lives above it. */
  __stack_top = 0x80000000 + RAM_SIZE - 16;
  __heap_end = __stack_top - STACK_BYTES;
  _end = __heap_start;

  /* The runner reads these to build the machine, so the ELF carries its own configuration. */
  __lx_ram_size = RAM_SIZE;
  __lx_text_end = TEXT_END;

  /DISCARD/ : { *(.note .note.*) *(.comment) }
}

ASSERT(SIZEOF(.text) <= TEXT_END, "code overflows the ROM region")
ASSERT(__heap_start + HEAP_BYTES <= __heap_end - 1024, "writable RAM overflows into the stack")
