/* linuxformer (LX machine) layout for Doom.
 *
 * The machine is one contiguous image of RAM at 0x80000000, ``ram_size`` bytes. The prefix below
 * ``text_end`` is fetched as code from the read-only copy, and -- this is the LX machine's one
 * constraint on a Doom port -- a *data* access below text_end traps, so only .text can live
 * there. Everything Doom reads as data, the 4 MB WAD included, sits above text_end.
 *
 *   [0x80000000, +TEXT_END)              .text (code, fetched from the ROM copy)
 *   [0x80000000 + TEXT_END, +RAM_SIZE)   .rodata, .data, .tdata/.tbss, .bss, heap, stack
 *
 * TEXT_END and RAM_SIZE must match lx.core.Config(text_end=..., ram_size=...) for the circuit
 * and the runner that loads this ELF.
 */
ENTRY(_start)

TEXT_END = 0x40000;    /* 256 KB of code */
RAM_SIZE = 0x600000;   /* 6 MB: the smallest image E1M1 loads in. Every MB of RAM costs the
                        * chip ~4 layers, so this is also 29% cheaper per token than 8 MB. */

MEMORY
{
  ROM (rx) : ORIGIN = 0x80000000,            LENGTH = TEXT_END
  RAM (rw) : ORIGIN = 0x80000000 + TEXT_END, LENGTH = RAM_SIZE - TEXT_END
}

SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM

  .rodata : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM

  /* 96 KB of stack at the top of the address space; Doom's zone is the static one in
     port/src/i_system.c, so the heap above __heap_start stays unused. */
  __stack_top = 0x80000000 + RAM_SIZE - 16;
  __heap_end = __stack_top - 0x18000;
  _end = __heap_start;

  /DISCARD/ : { *(.note .note.*) *(.comment) }
}

ASSERT(SIZEOF(.text) <= TEXT_END, "code overflows the ROM region")
ASSERT(__heap_start < __heap_end, "writable RAM overflows into the stack")
