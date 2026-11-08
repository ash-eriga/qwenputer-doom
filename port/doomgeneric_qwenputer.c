// Doom on the Qwenputer RV32 machine (the CPU compiled into Qwen3.5-4B's weights).
//
// Platform layer for doomgeneric. The machine has:
//   - no timer   -> the clock is the frame counter: one rendered frame = one 35 Hz tic
//   - no files   -> the WAD is served from read-only data (DROM) by w_file_qwenputer.c
//   - no keyboard-> a scripted key sequence (title -> menu -> new game),
//                   then a looping patrol; a demo on/off cell and a
//                   key-state cell let the runner hand control to a human
//   - putchar    -> ecall a7=1 (the machine's only output), wired up in the rv env's sys.c
//
// The framebuffer lives in guest RAM. The runner reads it (and qwen_frame_count)
// out of the machine state to display frames; nothing about the display is
// computed on the host -- the guest CPU renders every pixel.

#include <stdio.h>
#include <string.h>

#include "doomgeneric.h"
#include "doomkeys.h"

// ---------------------------------------------------------------- framebuffer

static pixel_t qwen_screen[DOOMGENERIC_RESX * DOOMGENERIC_RESY];
// DG_ScreenBuffer itself is defined by doomgeneric.c; DG_Init points it here.

// Polled by the runner: number of frames the guest has rendered so far.
volatile unsigned int qwen_frame_count = 0;

// Key-state cell the runner may write between steps (bit set = key held).
// Bit layout matches the scripted keys below: 0 up, 1 down, 2 left, 3 right,
// 4 fire, 5 use, 6 enter, 7 escape.
volatile unsigned int qwen_key_state = 0;

// Demo on/off, also poked by the runner: 1 = the demo (intro script, then
// the looping patrol) drives; 0 = the demo lets go — keys it was holding
// are released, script events already due are drained so they never fire
// late — and only qwen_key_state (a human at the controls) is read.
#ifdef QWEN_SCHEDULE_TABLE
volatile unsigned int qwen_demo_enabled = 0;  // the baked schedule drives from frame 0
#else
volatile unsigned int qwen_demo_enabled = 1;
#endif

#ifdef QWEN_SCHEDULE_TABLE
// The in-weights build: the schedule is baked into the guest (one entry per mask change,
// {0,0} terminates), applied by the guest itself — no host input exists after GO. The
// runner-poke rule is reproduced exactly: when the frame counter reads N, the mask becomes
// the last entry with frame <= N, so the next DG_GetKey (the next frame's input) sees it.
typedef struct { unsigned int frame; unsigned int mask; } qwen_mask_event_t;
static const qwen_mask_event_t qwen_schedule[] = {
#include "qwen_schedule.inc"
};
static unsigned int qwen_sched_pos = 0;

static void qwen_apply_schedule(void)
{
    while (qwen_schedule[qwen_sched_pos].frame &&
           qwen_frame_count + 1 >= qwen_schedule[qwen_sched_pos].frame)
        qwen_key_state = qwen_schedule[qwen_sched_pos++].mask;
}
#else
static void qwen_apply_schedule(void) {}
#endif

void DG_Init(void)
{
    DG_ScreenBuffer = qwen_screen;
    printf("qwenputer-doom: framebuffer %dx%d at %p\n",
           DOOMGENERIC_RESX, DOOMGENERIC_RESY, (void *)qwen_screen);
}

// ---------------------------------------------------------------- clock
// One frame = one tic (1000/35 ms). DG_SleepMs just advances the same clock,
// so the game runs exactly as fast as the CPU can render -- on the in-weights
// circuit that is the honest speed, whatever it turns out to be.

static uint32_t qwen_ticks_ms = 0;

uint32_t DG_GetTicksMs(void)
{
    return qwen_ticks_ms;
}

void DG_SleepMs(uint32_t ms)
{
    qwen_ticks_ms += ms;
}

void DG_DrawFrame(void)
{
    qwen_frame_count++;
#ifdef QWEN_SCHEDULE_TABLE
    qwen_apply_schedule();  // exactly the runner's poke point: the counter's store
#endif
    qwen_ticks_ms += 1000 / 35;
}

// ---------------------------------------------------------------- input
// Scripted key events, keyed by frame number: {frame, key, pressed}.
// Tune the frame numbers against the frame dumps from the qemu shakeout.

typedef struct { unsigned int frame; unsigned char key; int pressed; } qwen_key_event_t;

static const qwen_key_event_t qwen_script[] = {
    {  40, KEY_ENTER,     1 }, {  42, KEY_ENTER,     0 },  // title -> main menu
    {  50, KEY_ENTER,     1 }, {  52, KEY_ENTER,     0 },  // New Game
    {  60, KEY_ENTER,     1 }, {  62, KEY_ENTER,     0 },  // episode (shareware: Knee-Deep)
    {  70, KEY_ENTER,     1 }, {  72, KEY_ENTER,     0 },  // skill
    { 120, KEY_UPARROW,   1 },                            // walk forward
    { 200, KEY_FIRE,      1 }, { 206, KEY_FIRE,      0 },
    { 260, KEY_RIGHTARROW,1 },
    { 320, KEY_RIGHTARROW,0 },
    { 400, KEY_UPARROW,   0 },
    { 0, 0, 0 },
};

static unsigned int qwen_script_pos = 0;
static int qwen_live_key = 0;  // script index of a live (runner-poked) key being reported

// The patrol: once the intro script has played, the demo keeps the marine
// moving with this looping pattern (relative frames, cycle = 560). It is a
// patrol, not a player: forward, shoot, turn, back up, turn, push on
// doors/walls with USE, shoot again.
static const qwen_key_event_t qwen_patrol[] = {
    {   0, KEY_UPARROW,   1 },
    {  90, KEY_FIRE,      1 }, {  96, KEY_FIRE,      0 },
    { 102, KEY_FIRE,      1 }, { 108, KEY_FIRE,      0 },
    { 150, KEY_RIGHTARROW,1 }, { 185, KEY_RIGHTARROW,0 },
    { 300, KEY_UPARROW,   0 }, { 300, KEY_DOWNARROW, 1 },
    { 350, KEY_DOWNARROW, 0 },
    { 360, KEY_LEFTARROW, 1 }, { 395, KEY_LEFTARROW, 0 },
    { 420, KEY_UPARROW,   1 },
    { 470, KEY_USE,       1 }, { 476, KEY_USE,       0 },
    { 500, KEY_FIRE,      1 }, { 506, KEY_FIRE,      0 },
    { 550, KEY_UPARROW,   0 },
    { 0, 0, 0 },
};
#define QWEN_PATROL_CYCLE 560

static const unsigned char keymap[8] = {
    KEY_UPARROW, KEY_DOWNARROW, KEY_LEFTARROW, KEY_RIGHTARROW,
    KEY_FIRE, KEY_USE, KEY_ENTER, KEY_ESCAPE,
};

static unsigned int demo_held = 0;      // keymap bits the demo is holding down
static unsigned int patrol_epoch = 0;   // frame the current patrol cycle began
static unsigned int patrol_pos = 0;
static unsigned int patrol_last = 0;
static int patrol_active = 0;

static int demo_event(int *pressed, unsigned char *key,
                      unsigned char k, int p)
{
    int bit = 0;
    while (bit < 8 && keymap[bit] != k) bit++;
    if (bit < 8)
    {
        if (p) demo_held |= (1u << bit);
        else   demo_held &= ~(1u << bit);
    }
    *key = k;
    *pressed = p;
    return 1;
}

int DG_GetKey(int *pressed, unsigned char *key)
{
    qwen_apply_schedule();
    if (!qwen_demo_enabled)
    {
        // Demo off: release anything the demo was holding, one key per
        // call, and drain intro events already due so they can never
        // fire if the demo is switched back on later.
        if (demo_held)
        {
            int bit = 0;
            while (!(demo_held & (1u << bit))) bit++;
            demo_held &= ~(1u << bit);
            *key = keymap[bit];
            *pressed = 0;
            return 1;
        }
        while (qwen_script[qwen_script_pos].frame != 0 &&
               qwen_frame_count >= qwen_script[qwen_script_pos].frame)
            qwen_script_pos++;
        patrol_active = 0;
    }
    else if (qwen_script[qwen_script_pos].frame != 0)
    {
        // Intro script first.
        if (qwen_frame_count >= qwen_script[qwen_script_pos].frame)
        {
            unsigned char k = qwen_script[qwen_script_pos].key;
            int p = qwen_script[qwen_script_pos].pressed;
            qwen_script_pos++;
            return demo_event(pressed, key, k, p);
        }
    }
    else
    {
        // Intro done: the patrol loops.
        if (!patrol_active)
        {
            patrol_active = 1;
            patrol_epoch = qwen_frame_count;
            patrol_pos = 0;
            patrol_last = 0;
        }
        unsigned int rel = (qwen_frame_count - patrol_epoch) % QWEN_PATROL_CYCLE;
        if (rel < patrol_last) patrol_pos = 0;   // wrapped into a new cycle
        patrol_last = rel;
        // (the terminator's frame is 0, a real offset — bound on pos)
        if (patrol_pos < (sizeof(qwen_patrol) / sizeof(qwen_patrol[0]) - 1) &&
            rel >= qwen_patrol[patrol_pos].frame)
        {
            unsigned char k = qwen_patrol[patrol_pos].key;
            int p = qwen_patrol[patrol_pos].pressed;
            patrol_pos++;
            return demo_event(pressed, key, k, p);
        }
    }

    // Runner-poked keys: report transitions of qwen_key_state.
    static unsigned int reported_state = 0;
    unsigned int changed = qwen_key_state ^ reported_state;
    if (changed)
    {
        int bit = 0;
        while (!(changed & (1u << bit))) bit++;
        reported_state ^= (1u << bit);
        *key = keymap[bit];
        *pressed = (qwen_key_state >> bit) & 1;
        return 1;
    }

    return 0;
}

void DG_SetWindowTitle(const char *title)
{
    printf("qwenputer-doom: %s\n", title);
}

// ---------------------------------------------------------------- main

int main(void)
{
    char *argv[] = {
        "doom",
        "-iwad", "DOOM.WAD",
        "-nosound",
        "-nomusic",
#ifdef LF_WARP
        // The LX build starts straight into a level: the menus cost ~200 frames of the guest and
        // their outcome depends on how fast ENTER taps arrive, so a scripted playthrough is only
        // reproducible without them.
        "-warp", "1", LF_MAP,
        "-skill", LF_SKILL,
#endif
        NULL,
    };
    // doomgeneric_Create takes argc, and anything past it is ignored -- count the array instead of
    // hardcoding the number, or the extra arguments are silently dropped.
    int argc = 0;
    while (argv[argc]) argc++;
    doomgeneric_Create(argc, argv);
    for (;;)
        doomgeneric_Tick();
    return 0;
}
