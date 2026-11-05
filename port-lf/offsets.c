/* Prints the RV32 (ilp32) offsets and sizes of the Doom structures the trace reads.
 * Host compilers are 64-bit here, so offsetof() on the host is wrong: pointers are 8 bytes
 * there and 4 on the target. This program is built for the target and run inside fastemu-lf,
 * so what it prints is exactly what the trace must use.
 */
#include <stddef.h>
#include "rt.h"
#include "doomdef.h"
#include "d_player.h"
#include "p_mobj.h"
#include "p_tick.h"

static void put(const char *s)
{
    while (*s) lf_putchar(*s++);
}

static void putu(unsigned v)
{
    char b[12];
    int i = 0;
    if (!v) { lf_putchar('0'); return; }
    while (v) { b[i++] = (char)('0' + v % 10); v /= 10; }
    while (i--) lf_putchar(b[i]);
}

static void kv(const char *k, unsigned v) { put(k); put("="); putu(v); put("\n"); }

int main(void)
{
    kv("sizeof_thinker_t", sizeof(thinker_t));
    kv("sizeof_mobj_t", sizeof(mobj_t));
    kv("sizeof_player_t", sizeof(player_t));
    kv("thinker_prev", offsetof(thinker_t, prev));
    kv("thinker_next", offsetof(thinker_t, next));
    kv("thinker_function", offsetof(thinker_t, function));
    kv("mo_x", offsetof(mobj_t, x));
    kv("mo_y", offsetof(mobj_t, y));
    kv("mo_z", offsetof(mobj_t, z));
    kv("mo_angle", offsetof(mobj_t, angle));
    kv("mo_type", offsetof(mobj_t, type));
    kv("mo_health", offsetof(mobj_t, health));
    kv("mo_flags", offsetof(mobj_t, flags));
    kv("pl_mo", offsetof(player_t, mo));
    kv("pl_health", offsetof(player_t, health));
    kv("pl_killcount", offsetof(player_t, killcount));
    kv("pl_ammo", offsetof(player_t, ammo));
    kv("pl_readyweapon", offsetof(player_t, readyweapon));
    lf_halt(0);
    return 0;
}
