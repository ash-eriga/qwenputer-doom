# Minimal C runtime for the linuxformer machine: sp is the top of RAM, tp points at the TLS block
# (picolibc's errno is thread-local), main's return value halts the trace through SYSCON.
    .section .text.init
    .globl _start
_start:
    .option push
    .option norelax
    la sp, __stack_top
    la tp, __tls_base
    .option pop
    call main
    call exit
1:  j 1b
