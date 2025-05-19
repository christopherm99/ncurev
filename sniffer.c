#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/syscall.h>
#include <linux/limits.h>
#include <linux/ioctl.h>
#include <unistd.h>

#include "uthash.h"

#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include "nvos.h"
#include "nv-unix-nvos-params-wrappers.h"

struct fd_info {
  int fd;
  char name[PATH_MAX];
  UT_hash_handle hh;
};

struct fd_info *fd_map = NULL;
int mem_fd;

void add_fd(int fd, const char *name) {
  struct fd_info *info;

  HASH_FIND_INT(fd_map, &fd, info);
  if (info == NULL) {
    info = malloc(sizeof(struct fd_info));
    info->fd = fd;
    HASH_ADD_INT(fd_map, fd, info);
  }
  strcpy(info->name, name);
}

char *get_fd_name(int fd) {
  struct fd_info *info;

  HASH_FIND_INT(fd_map, &fd, info);
  if (info != NULL) {
    return info->name;
  }
  return NULL;
}

char *copy_string(unsigned long addr) {
  size_t cap = 32;
  size_t off = 0;
  char *buf;

  buf = malloc(cap);

  lseek(mem_fd, addr, SEEK_SET);

  while (1) {
    for (; off < cap; off++) {
      if (read(mem_fd, buf + off, 1) != 1) {
        perror("read mem");
        exit(-1);
      }
      if (*(buf + off) == '\0') return buf;
    }
    cap *= 2;
    buf = realloc(buf, cap);
  }
  return NULL;
}

void *copy_mem(unsigned long addr, size_t size) {
  char *buf = malloc(size);
  lseek(mem_fd, addr, SEEK_SET);
  if (read(mem_fd, buf, size) != size) {
    perror("read mem");
    exit(-1);
  }
  return buf;
}

void print_ioctl(int fd, unsigned long request, unsigned long arg) {
  char *name = get_fd_name(fd);
  unsigned long dir, type, nr, size;

  dir = _IOC_DIR(request);
  type = _IOC_TYPE(request);
  nr = _IOC_NR(request);
  size = _IOC_SIZE(request);
  printf("ioctl %16s %5s ", name ? name : "unknown", (dir & _IOC_READ) && (dir & _IOC_WRITE) ? "_IORW" : (dir & _IOC_READ) ? "_IOR" : (dir & _IOC_WRITE) ? "_IOW" : "_IO");
  if (type == NV_IOCTL_MAGIC) {
    void *data = copy_mem(arg, size);
    printf("NV_IOCTL_MAGIC ");
    switch (nr) {
    case NV_ESC_CARD_INFO: puts("NV_ESC_CARD_INFO"); break;
    case NV_ESC_REGISTER_FD: puts("NV_ESC_REGISTER_FD"); break;
    case NV_ESC_ALLOC_OS_EVENT: puts("NV_ESC_ALLOC_OS_EVENT"); break;
    case NV_ESC_FREE_OS_EVENT: puts("NV_ESC_FREE_OS_EVENT"); break;
    case NV_ESC_STATUS_CODE: puts("NV_ESC_STATUS_CODE"); break;
    case NV_ESC_CHECK_VERSION_STR: puts("NV_ESC_CHECK_VERSION_STR"); break;
    case NV_ESC_IOCTL_XFER_CMD: puts("NV_ESC_IOCTL_XFER_CMD"); break;
    case NV_ESC_ATTACH_GPUS_TO_FD: puts("NV_ESC_ATTACH_GPUS_TO_FD"); break;
    case NV_ESC_QUERY_DEVICE_INTR: puts("NV_ESC_QUERY_DEVICE_INTR"); break;
    case NV_ESC_SYS_PARAMS: puts("NV_ESC_SYS_PARAMS"); break;
    case NV_ESC_EXPORT_TO_DMABUF_FD: puts("NV_ESC_EXPORT_TO_DMABUF_FD"); break;
    case NV_ESC_WAIT_OPEN_COMPLETE: puts("NV_ESC_WAIT_OPEN_COMPLETE"); break;
    case NV_ESC_RM_ALLOC_MEMORY: {
      NVOS02_PARAMETERS *p = (NVOS02_PARAMETERS *)data;
      printf("NV_ESC_RM_ALLOC hRoot: %x pMemory: %p limit: %llx\n", p->hRoot, p->pMemory, p->limit);
    } break;
    case NV_ESC_RM_FREE: puts("NV_ESC_RM_FREE"); break;
    case NV_ESC_RM_CONTROL: {
      NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)data;
      printf("NV_ESC_RM_CONTROL ");
      #define cmd(name) case name: printf(#name); break;
      switch (p->cmd) {
        default: printf("unknown command %x", p->cmd); break;
      }
      #undef cmd
      printf(" client: %x object: %x params: %p flags: %x status 0x%x\n", p->hClient, p->hObject, p->params, p->flags, p->status);
    } break;
    case NV_ESC_RM_ALLOC: puts("NV_ESC_RM_ALLOC"); break;
    case NV_ESC_RM_MAP_MEMORY: {
      nv_ioctl_nvos33_parameters_with_fd *pfd = (nv_ioctl_nvos33_parameters_with_fd *)data;
      NVOS33_PARAMETERS *p = (NVOS33_PARAMETERS *)data;
      printf("NV_ESC_RM_MAP_MEMORY hClient: %x hDevice: %x hMemory: %x pLinearAddress: %p offset: %llx length: %llx status %x flags %x file: %s (fd=%d)\n",
        p->hClient, p->hDevice, p->hMemory, p->pLinearAddress, p->offset, p->length, p->status, p->flags,
        get_fd_name(pfd->fd), pfd->fd);
    } break;
    case NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO: puts("NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO"); break;
    default: printf("unknown ioctl 0x%lx\n", nr); break;
    }
  } else {
    puts("...");
  }
}

int main(int argc, char **argv) {
  pid_t pid;
  char mem_path[PATH_MAX];

  if (argc != 2) {
    printf("Usage: %s <pid>\n", argv[0]);
    return -1;
  }

  pid = atoi(argv[1]);

  printf("Attaching to process %d...\n", pid);

  if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) == -1) {
    perror("ptrace attach");
    return -1;
  }

  kill(pid, SIGCONT);
  waitpid(pid, NULL, 0);

  snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", pid);
  mem_fd = open(mem_path, O_RDWR);

  if (ptrace(PTRACE_SETOPTIONS, pid, NULL, PTRACE_O_TRACESYSGOOD) == -1) {
    perror("ptrace setoptions");
    return -1;
  }

  while (1) {
    int status;

    if (ptrace(PTRACE_SYSCALL, pid, NULL, NULL) == -1) {
      perror("ptrace syscall");
      break;
    }

    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
      printf("Process %d exited\n", pid);
      break;
    }

    if (WIFSIGNALED(status)) {
      printf("Process %d killed by signal %d\n", pid, WTERMSIG(status));
      break;
    }

    if (WIFSTOPPED(status) && WSTOPSIG(status) == (SIGTRAP | 0x80)) {
      struct user_regs_struct regs;
      static int in = 1;

      if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
        perror("ptrace getregs");
        break;
      }

      if (in) {
        in = 0;
      } else {
        switch (regs.orig_rax) {
        case SYS_open:
          add_fd((int)regs.rax, copy_string(regs.rdi));
          break;
        case SYS_openat:
          add_fd((int)regs.rax, copy_string(regs.rsi));
          break;
        case SYS_ioctl:
          print_ioctl((int)regs.rdi, regs.rsi, regs.rdx);
          break;
        }
        in = 1;
      }
    }
  }

  ptrace(PTRACE_DETACH, pid, NULL, NULL);
  return 0;
}

