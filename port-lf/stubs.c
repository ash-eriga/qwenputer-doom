// POSIX stubs for the linuxformer machine: there is no filesystem.
// Files fail to open (config files and savegames are optional in the engine); stdout goes to the
// UART, one BYTE token per byte.

#include <errno.h>
#include <stddef.h>
#include <sys/types.h>

#include "rt.h"

int open(const char *path, int flags, ...)
{
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
            lf_putchar(p[i]);
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

// No process model. doomgeneric's only fork/exec use is the zenity error box in i_system.c;
// picolibc's system() pulls these in, so they must exist and fail cleanly.
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
