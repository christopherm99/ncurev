#include <fcntl.h>
#include <linux/limits.h>
#include <linux/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>


#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0000/ctrl0000client.h"
#include "ctrl/ctrl0000/ctrl0000syncgpuboost.h"
#include "ctrl/ctrl0000/ctrl0000system.h"
#include "ctrl/ctrl0080.h"
#include "ctrl/ctrl2080.h"
#include "ctrl/ctrl83de.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrlb0cc.h"
#include "ctrl/ctrlc36f.h"
#include "ctrl/ctrlcb33.h"
#include "nvos.h"
#include "nv_escape.h"
#include "nv-ioctl-numbers.h"
#include "nv-unix-nvos-params-wrappers.h"

#include "pprint.h"
#include "uthash.h"

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
  static int idx = 0;

  dir = _IOC_DIR(request);
  type = _IOC_TYPE(request);
  nr = _IOC_NR(request);
  size = _IOC_SIZE(request);
  printf("ioctl %5d %16s %5s ", idx++, name ? name : "unknown", (dir & _IOC_READ) && (dir & _IOC_WRITE) ? "_IORW" : (dir & _IOC_READ) ? "_IOR" : (dir & _IOC_WRITE) ? "_IOW" : "_IO");
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
      #define cmd(name) case name: puts(#name); break
      #define pprint(name, val) pprint_##name(val);
      #define ppprint(name) case name: pprint_##name(copy_mem((unsigned long)p->params, p->paramsSize)); break
      switch (p->cmd) {
        cmd(NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION);
        cmd(NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS);
        // TODO: these should show status values
        cmd(NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG);
        cmd(NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS);
        cmd(NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS);
        ppprint(NV0000_CTRL_CMD_GPU_ATTACH_IDS);
        ppprint(NV0000_CTRL_CMD_GPU_GET_ID_INFO);
        ppprint(NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS);
        ppprint(NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2);
        ppprint(NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID);
        ppprint(NV0000_CTRL_CMD_GPU_GET_PROBED_IDS);
        ppprint(NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE);
        ppprint(NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE);
        ppprint(NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY);
        ppprint(NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO);
        ppprint(NV0000_CTRL_CMD_SYSTEM_GET_FEATURES);
        ppprint(NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX);
        ppprint(NV2080_CTRL_CMD_GR_GET_TPC_MASK);
        ppprint(NV2080_CTRL_CMD_FB_GET_INFO_V2);
        ppprint(NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS);
        ppprint(NV2080_CTRL_CMD_BUS_GET_INFO_V2);
        ppprint(NV2080_CTRL_CMD_MC_GET_ARCH_INFO);
        ppprint(NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE);
        ppprint(NV2080_CTRL_CMD_GPU_GET_GID_INFO);
        ppprint(NV2080_CTRL_CMD_GSP_GET_FEATURES);
        ppprint(NV2080_CTRL_CMD_CE_GET_ALL_CAPS);
        ppprint(NV2080_CTRL_CMD_BUS_GET_C2C_INFO);
        ppprint(NV2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS);
        ppprint(NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO);
        ppprint(NV2080_CTRL_CMD_BUS_GET_PCI_INFO);
        ppprint(NV2080_CTRL_CMD_GR_GET_GPC_MASK);
        ppprint(NV2080_CTRL_CMD_GR_GET_CAPS_V2);
        ppprint(NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER);
        ppprint(NV2080_CTRL_CMD_GR_GET_INFO);
        ppprint(NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS);
        ppprint(NV2080_CTRL_CMD_GPU_GET_ENGINES_V2);
        ppprint(NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES);
        ppprint(NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO);
        ppprint(NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING);
        ppprint(NV2080_CTRL_CMD_GPU_GET_NAME_STRING);
        ppprint(NV2080_CTRL_CMD_GPU_GET_INFO_V2);
        ppprint(NV2080_CTRL_CMD_PERF_BOOST);
        ppprint(NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE);
        ppprint(NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST);
        ppprint(NV0080_CTRL_CMD_HOST_GET_CAPS_V2);
        ppprint(NV0080_CTRL_CMD_FB_GET_CAPS_V2);
        ppprint(NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2);
        ppprint(NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE);
        ppprint(NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES);
        ppprint(NV0080_CTRL_CMD_PERF_CUDA_LIMIT_SET_CONTROL);
        ppprint(NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK);
        ppprint(NVC36F_CTRL_GET_CLASS_ENGINEID);
        ppprint(NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN);
        ppprint(NV_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES);
        ppprint(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE);
        ppprint(NVA06C_CTRL_CMD_SET_TIMESLICE);
        // *** NCU-related ***
        cmd(NVB0CC_CTRL_CMD_BIND_PM_RESOURCES);
        ppprint(NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT);
        ppprint(NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES);
        ppprint(NVB0CC_CTRL_CMD_POWER_RELEASE_FEATURES);
        ppprint(NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM);
        ppprint(NVB0CC_CTRL_CMD_FREE_PMA_STREAM);
        ppprint(NVB0CC_CTRL_CMD_RESERVE_PM_AREA_SMPC);
        ppprint(NVB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY);
        ppprint(NVB0CC_CTRL_CMD_GET_TOTAL_HS_CREDITS);
        case NV2080_CTRL_CMD_GPU_EXEC_REG_OPS: {
          NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS *params = (NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS *)copy_mem((unsigned long)p->params, p->paramsSize);
          NV2080_CTRL_GPU_REG_OP *ops = (NV2080_CTRL_GPU_REG_OP *)copy_mem((unsigned long)params->regOps, params->regOpCount * sizeof(NV2080_CTRL_GPU_REG_OP));
          printf("NV2080_CTRL_CMD_GPU_EXEC_REG_OPS regOpCount: %d\n", params->regOpCount);
          for (int i = 0; i < params->regOpCount; i++) {
            printf("  %d: ", i);
            pprint(NV2080_CTRL_GPU_REG_OP, ops + i);
          }
        } break;
        case NVB0CC_CTRL_CMD_EXEC_REG_OPS: {
          NVB0CC_CTRL_EXEC_REG_OPS_PARAMS *params = (NVB0CC_CTRL_EXEC_REG_OPS_PARAMS *)copy_mem((unsigned long)p->params, p->paramsSize);
          printf("NVB0CC_CTRL_CMD_EXEC_REG_OPS regOpCount: %d\n", params->regOpCount);
          for (int i = 0; i < params->regOpCount; i++) {
            printf("  %d: ", i);
            pprint(NV2080_CTRL_GPU_REG_OP, &params->regOps[i]);
          }
        } break;
        case NVB0CC_CTRL_CMD_SET_HS_CREDITS: {
          NVB0CC_CTRL_HS_CREDITS_PARAMS *params = (NVB0CC_CTRL_HS_CREDITS_PARAMS *)copy_mem((unsigned long)p->params, p->paramsSize);
          pprint(NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS, &params->statusInfo);
          for (int i = 0; i < params->numEntries; i++) {
            printf("  %d: ", i);
            pprint(NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO, &params->creditInfo[i]);
          }
        } break;
        default: printf("unknown command %x\n", p->cmd); break;
      }
      #undef cmd
      #undef pprint
      #undef ppprint
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

