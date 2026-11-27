    .section .rodata
    .globl qwen_wad_data
    .globl qwen_wad_length
    .balign 4
qwen_wad_data:
    .incbin "placeholder.wad"
qwen_wad_end:
    .balign 4
qwen_wad_length:
    .word qwen_wad_end - qwen_wad_data
