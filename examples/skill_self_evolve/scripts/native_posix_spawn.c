#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* Some Python builds omit pidfd APIs; kernel constants come from C headers. */
int native_pidfd_open(int pid) {
    int fd = (int)syscall(SYS_pidfd_open, pid, 0);
    return fd < 0 ? -errno : fd;
}

int native_pidfd_signal(int fd, int number) {
    return syscall(SYS_pidfd_send_signal, fd, number, NULL, 0) < 0 ? errno : 0;
}

/* Keep glibc's opaque spawn structures on the C side of the ABI. */
int native_posix_spawn(const char *program, char *const argv[], char *const envp[],
                       const char *cwd, const int sources[4], int *pid_out) {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attrs;
    int copies[4] = {-1, -1, -1, -1};
    int error, actions_ready = 0, attrs_ready = 0;
    pid_t pid;
    sigset_t defaults, mask;

    /* All sources must survive every dup2, including sources originally 0..3. */
    for (int i = 0; i < 4; ++i) {
        copies[i] = fcntl(sources[i], F_DUPFD_CLOEXEC, 4);
        if (copies[i] < 0) { error = errno; goto done; }
    }
#define CHECK(expr) do { error = (expr); if (error) goto done; } while (0)
    CHECK(posix_spawn_file_actions_init(&actions));
    actions_ready = 1;
    CHECK(posix_spawnattr_init(&attrs));
    attrs_ready = 1;
    for (int i = 0; i < 4; ++i)
        CHECK(posix_spawn_file_actions_adddup2(&actions, copies[i], i));
    CHECK(posix_spawn_file_actions_addchdir_np(&actions, cwd));
    CHECK(posix_spawn_file_actions_addclosefrom_np(&actions, 4));
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
#ifdef SIGXFZ
    sigaddset(&defaults, SIGXFZ);
#endif
#ifdef SIGXFSZ
    sigaddset(&defaults, SIGXFSZ);
#endif
    sigemptyset(&mask);
    CHECK(posix_spawnattr_setsigdefault(&attrs, &defaults));
    CHECK(posix_spawnattr_setsigmask(&attrs, &mask));
    CHECK(posix_spawnattr_setflags(&attrs, POSIX_SPAWN_SETSID |
                                  POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK));
    CHECK(posix_spawn(&pid, program, &actions, &attrs, argv, envp));
    *pid_out = (int)pid;
done:
    if (attrs_ready) posix_spawnattr_destroy(&attrs);
    if (actions_ready) posix_spawn_file_actions_destroy(&actions);
    for (int i = 0; i < 4; ++i) if (copies[i] >= 0) close(copies[i]);
    return error;
}
