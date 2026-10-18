// Diagnostic LD_PRELOAD shim: present /proc mount tables with an explicit data=ordered on the root
// ext4 filesystem (which already runs in ordered mode, per /proc/fs/ext4/<dev>/options), so cuFile's
// journal-mode check can see it. Affects only the process it is preloaded into.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int is_table(const char *p) {
    return p && (!strcmp(p, "/proc/self/mountinfo") || !strcmp(p, "/proc/mounts") ||
                 !strcmp(p, "/proc/self/mounts") || !strcmp(p, "/etc/mtab") ||
                 (!strncmp(p, "/proc/", 6) && (strstr(p, "/mountinfo") || strstr(p, "/mounts"))));
}
static int patched_fd(const char *path) {
    static int (*real_open)(const char *, int, ...);
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");
    int in = real_open(path, O_RDONLY);
    if (in < 0) return in;
    size_t cap = 1 << 20, n = 0; char *buf = malloc(cap); ssize_t r;
    while ((r = read(in, buf + n, cap - n)) > 0) { n += r; if (n == cap) buf = realloc(buf, cap *= 2); }
    close(in);
    int out = memfd_create("mounts", 0);
    char *line = buf, *end = buf + n;
    while (line < end) {
        char *nl = memchr(line, '\n', end - line); size_t len = nl ? (size_t)(nl - line) : (size_t)(end - line);
        char tmp[8192]; if (len >= sizeof tmp - 32) len = sizeof tmp - 32;
        memcpy(tmp, line, len); tmp[len] = 0;
        // Root ext4 only: append data=ordered to the last options field (super options / mount options).
        if (strstr(tmp, " ext4 ") && (strstr(tmp, " / ") ) && !strstr(tmp, "data=")) {
            char *sp = strrchr(tmp, ' ');
            if (strstr(path, "mountinfo")) strcat(tmp, ",data=ordered");
            else if (sp) { // /proc/mounts: "dev / ext4 opts 0 0"
                char *f[6]; int k = 0; char *s = tmp, *save;
                char copy[8192]; strcpy(copy, tmp);
                for (char *t = strtok_r(copy, " ", &save); t && k < 6; t = strtok_r(NULL, " ", &save)) f[k++] = t;
                if (k == 6) snprintf(tmp, sizeof tmp, "%s %s %s %s,data=ordered %s %s", f[0], f[1], f[2], f[3], f[4], f[5]);
                (void)s;
            }
        }
        write(out, tmp, strlen(tmp)); write(out, "\n", 1);
        line += (nl ? (size_t)(nl - line) + 1 : len);
        if (!nl) break;
    }
    free(buf);
    lseek(out, 0, SEEK_SET);
    return out;
}
int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...);
    if (!real) real = dlsym(RTLD_NEXT, "open");
    if (is_table(path) && !(flags & (O_WRONLY | O_RDWR))) return patched_fd(path);
    va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
    return real(path, flags, mode);
}
int open64(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...);
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    if (is_table(path) && !(flags & (O_WRONLY | O_RDWR))) return patched_fd(path);
    va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
    return real(path, flags, mode);
}
int openat(int dir, const char *path, int flags, ...) {
    static int (*real)(int, const char *, int, ...);
    if (!real) real = dlsym(RTLD_NEXT, "openat");
    if (is_table(path) && !(flags & (O_WRONLY | O_RDWR))) return patched_fd(path);
    va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
    return real(dir, path, flags, mode);
}
FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = dlsym(RTLD_NEXT, "fopen");
    if (is_table(path) && mode && mode[0] == 'r' && !strchr(mode, '+')) { int fd = patched_fd(path); return fd < 0 ? NULL : fdopen(fd, "r"); }
    return real(path, mode);
}
FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = dlsym(RTLD_NEXT, "fopen64");
    if (is_table(path) && mode && mode[0] == 'r' && !strchr(mode, '+')) { int fd = patched_fd(path); return fd < 0 ? NULL : fdopen(fd, "r"); }
    return real(path, mode);
}
FILE *setmntent(const char *path, const char *mode) { return fopen(path, mode); }
