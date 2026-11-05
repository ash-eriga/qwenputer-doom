// POSIX stubs for the Qwenputer machine: there is no filesystem.
// Files simply fail to open (config files and savegames are skipped by the
// engine, which treats them as optional); stdout writes go to putchar.

#include <errno.h>
#include <stddef.h>
#include <sys/types.h>

// The machine's putchar: ecall a7=1 (same as the rv env's rt.h inline).
static void qwen_putchar(char c)
{
    register long r_a0 __asm__("a0") = (unsigned char)c;
    register long r_a1 __asm__("a1") = 0, r_a2 __asm__("a2") = 0;
    register long r_a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a2), "r"(r_a7) : "memory");
}

int open(const char *path, int flags, ...)
{
    // No filesystem -- but several engine probes (IWAD search, config load)
    // treat "cannot open" as fatal while the data itself is served by the
    // DROM WAD class. So opens succeed as a fake fd that reads EOF and
    // swallows writes. The WAD bytes never come through here.
    (void)path; (void)flags;
    return 3;
}

int close(int fd)
{
    (void)fd;
    return -1;
}

ssize_t read(int fd, void *buf, size_t count)
{
    (void)fd; (void)buf; (void)count;
    return 0;  // EOF
}

ssize_t write(int fd, const void *buf, size_t count)
{
    if (fd == 1 || fd == 2)
    {
        const char *p = (const char *)buf;
        for (size_t i = 0; i < count; i++)
            qwen_putchar(p[i]);
        return (ssize_t)count;
    }
    return -1;
}

off_t lseek(int fd, off_t offset, int whence)
{
    (void)fd; (void)offset; (void)whence;
    return -1;
}

int unlink(const char *path)
{
    (void)path;
    return -1;
}

int rename(const char *old_path, const char *new_path)
{
    (void)old_path; (void)new_path;
    return -1;
}

int mkdir(const char *path, mode_t mode)
{
    (void)path; (void)mode;
    return -1;
}

// No process model on the machine. doomgeneric's only fork/exec use is the
// zenity error box in i_system.c; picolibc's system() pulls these in, so they
// must exist and fail cleanly.
int fork(void)
{
    return -1;
}

int waitpid(int pid, int *status, int options)
{
    (void)pid; (void)status; (void)options;
    return -1;
}

int execve(const char *path, char *const argv[], char *const envp[])
{
    (void)path; (void)argv; (void)envp;
    return -1;
}
