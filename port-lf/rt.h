// Output and halt on the linuxformer RV32 machine (the LX machine).
//
// The machine has no ecall output: a store to the UART's THR (0x10000000) emits one BYTE token
// per byte, and a store to SYSCON (0x11100000) emits HALT. Both are ordinary stores the circuit
// compiles; nothing outside the model runs.
#pragma once

#include <stdint.h>

#define LF_UART_THR 0x10000000u
#define LF_UART_LSR 0x10000005u
#define LF_SYSCON   0x11100000u

static inline void lf_putchar(char c)
{
    *(volatile uint8_t *)LF_UART_THR = (uint8_t)c;
}

// The LSR reports "transmit empty, no input waiting" so the engine's probes never block.
// bit 5 = THR empty, bit 6 = transmitter empty.
static inline int lf_uart_ready(void)
{
    return (*(volatile uint8_t *)LF_UART_LSR & 0x60) == 0x60;
}

static inline void lf_halt(int code) __attribute__((noreturn));
static inline void lf_halt(int code)
{
    // The machine accepts 0x5555 (success) and 0x7777 (failure) at SYSCON.
    *(volatile uint32_t *)LF_SYSCON = (uint32_t)(code == 0 ? 0x5555 : 0x7777);
    for (;;)
        __asm__ volatile("nop");
}
