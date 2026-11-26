/* Small-memory LX layout for the 192x120 assembly raycaster.
 *
 * 24 KiB ROM-prefix + 72 KiB writable RAM = 96 KiB total LX image (RAM_SIZE=0x18000).
 * Keeps the emitted circuit at 14 layers (38 heads, max_heads=10) — the same layer count
 * as the 96x60 small build, vs 55 layers / 768 heads for 6 MiB DoomGeneric.
 */
ENTRY(_start)

TEXT_END = 0x6000;
RAM_SIZE = 0x18000;
__lx_text_end = TEXT_END;
__lx_ram_size = RAM_SIZE;

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

  __stack_top = 0x80000000 + RAM_SIZE - 16;
  __heap_end = __stack_top - 0x1000;
  _end = __heap_start;

  /DISCARD/ : { *(.note .note.*) *(.comment) }
}

ASSERT(SIZEOF(.text) <= TEXT_END, "code overflows the 192x120 ROM region")
ASSERT(__heap_start < __heap_end, "writable RAM overflows into the stack")
