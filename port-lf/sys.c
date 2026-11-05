// picolibc stdio and exit on the linuxformer RV32 machine.
#include <stdio.h>
#include <stdlib.h>

#include "rt.h"

static int lf_putc(char c, FILE *f)
{
    (void)f;
    lf_putchar(c);
    return (unsigned char)c;
}

static FILE __stdio = FDEV_SETUP_STREAM(lf_putc, NULL, NULL, _FDEV_SETUP_WRITE);
FILE *const stdin = &__stdio;
FILE *const stdout = &__stdio;
FILE *const stderr = &__stdio;

void _exit(int code)
{
    lf_halt(code);
}
