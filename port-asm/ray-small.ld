/* Small-memory LX layout for the assembly cost probe.
 *
 * RAM heads are the expensive part of the circuit: 2048 words per key head. This image uses
 * 8 KiB of ROM-prefix and 32 KiB total RAM, so only RAM blocks 1..3 exist instead of the
 * hundreds needed by the 6 MiB Doom image.
 */
ENTRY(_start)

TEXT_END = 0x2000;
RAM_SIZE = 0x8000;
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

ASSERT(SIZEOF(.text) <= TEXT_END, "code overflows the small ROM region")
ASSERT(__heap_start < __heap_end, "writable RAM overflows into the stack")
