/*
 *
 * Copyright (C) 2026 Diljit Singh
 *
 * This file is part of Nvtop.
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nvtop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nvtop.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// Apple GPUs under the Linux asahi DRM driver (Apple silicon Macs). Device-wide figures come from the driver's
// agx_stats and gpu_memory sysfs files, per-process figures from the standard DRM fdinfo keys.

#include "nvtop/device_discovery.h"
#include "nvtop/extract_gpuinfo_common.h"
#include "nvtop/extract_processinfo_fdinfo.h"
#include "nvtop/time.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uthash.h>

#define HASH_FIND_CLIENT(head, key_ptr, out_ptr) HASH_FIND(hh, head, key_ptr, sizeof(struct unique_cache_id), out_ptr)
#define HASH_ADD_CLIENT(head, in_ptr) HASH_ADD(hh, head, client_id, sizeof(struct unique_cache_id), in_ptr)

enum asahi_engine { asahi_vertex = 0, asahi_fragment, asahi_compute, asahi_engine_count };

struct __attribute__((__packed__)) unique_cache_id {
  unsigned client_id;
  pid_t pid;
};

struct asahi_process_info_cache {
  struct unique_cache_id client_id;
  uint64_t engine_ns[asahi_engine_count];
  bool engine_valid[asahi_engine_count];
  nvtop_time last_measurement_tstamp;
  UT_hash_handle hh;
};

struct gpu_info_asahi {
  struct gpu_info base;
  struct nvtop_device *card_device;
  struct nvtop_device *driver_device;
  char stats_path[PATH_MAX];
  char memory_path[PATH_MAX];
  // Device busy time (agx_stats busy_ns) at the previous refresh, for a utilisation over the refresh interval.
  bool busy_valid;
  uint64_t busy_ns;
  nvtop_time busy_tstamp;
  // Highest engine frequency any client's fdinfo reported (drm-maxfreq-*), in MHz.
  unsigned maxfreq_mhz;
  struct asahi_process_info_cache *last_update_process_cache, *current_update_process_cache;
};

static bool gpuinfo_asahi_init(void);
static void gpuinfo_asahi_shutdown(void);
static const char *gpuinfo_asahi_last_error_string(void);
static bool gpuinfo_asahi_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_asahi_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_asahi_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_asahi_get_running_processes(struct gpu_info *_gpu_info);

struct gpu_vendor gpu_vendor_asahi = {
    .init = gpuinfo_asahi_init,
    .shutdown = gpuinfo_asahi_shutdown,
    .last_error_string = gpuinfo_asahi_last_error_string,
    .get_device_handles = gpuinfo_asahi_get_device_handles,
    .populate_static_info = gpuinfo_asahi_populate_static_info,
    .refresh_dynamic_info = gpuinfo_asahi_refresh_dynamic_info,
    .refresh_running_processes = gpuinfo_asahi_get_running_processes,
    .name = "asahi",
};

static unsigned asahi_gpu_count;
static struct gpu_info_asahi *gpu_infos;

__attribute__((constructor)) static void init_extract_gpuinfo_asahi(void) { register_gpu_vendor(&gpu_vendor_asahi); }

static bool gpuinfo_asahi_init(void) { return true; }

static void gpuinfo_asahi_shutdown(void) {
  for (unsigned i = 0; i < asahi_gpu_count; ++i) {
    struct gpu_info_asahi *current = &gpu_infos[i];
    struct asahi_process_info_cache *entry, *tmp;
    HASH_ITER(hh, current->last_update_process_cache, entry, tmp) {
      HASH_DEL(current->last_update_process_cache, entry);
      free(entry);
    }
    HASH_ITER(hh, current->current_update_process_cache, entry, tmp) {
      HASH_DEL(current->current_update_process_cache, entry);
      free(entry);
    }
    nvtop_device_unref(current->card_device);
    nvtop_device_unref(current->driver_device);
  }
  free(gpu_infos);
  gpu_infos = NULL;
  asahi_gpu_count = 0;
}

static const char *gpuinfo_asahi_last_error_string(void) { return "Err"; }

// The kernel driver is "asahi"; some kernels name it "aurora".
static bool is_asahi_driver(const char *name) { return !strcmp(name, "asahi") || !strcmp(name, "aurora"); }

// "<number>[ <unit>]" with unit B, KiB, MiB or GiB (Documentation/gpu/drm-usage-stats.rst).
static bool parse_fdinfo_size(const char *val, uint64_t *bytes) {
  char *end;
  uint64_t value = strtoull(val, &end, 10);
  if (end == val)
    return false;
  if (!*end || !strcmp(end, " B"))
    *bytes = value;
  else if (!strcmp(end, " KiB") || !strcmp(end, " kB"))
    *bytes = value << 10;
  else if (!strcmp(end, " MiB"))
    *bytes = value << 20;
  else if (!strcmp(end, " GiB"))
    *bytes = value << 30;
  else
    return false;
  return true;
}

// "<number> Hz|KHz|MHz" -> MHz.
static bool parse_fdinfo_frequency_mhz(const char *val, unsigned *mhz) {
  char *end;
  uint64_t value = strtoull(val, &end, 10);
  if (end == val)
    return false;
  if (!strcmp(end, " Hz"))
    *mhz = value / 1000000;
  else if (!strcmp(end, " KHz") || !strcmp(end, " kHz"))
    *mhz = value / 1000;
  else if (!strcmp(end, " MHz"))
    *mhz = value;
  else
    return false;
  return true;
}

static bool parse_drm_fdinfo_asahi(struct gpu_info *info, FILE *fdinfo_file, struct gpu_process *process_info) {
  static const char *const engine_keys[asahi_engine_count] = {"drm-engine-vertex", "drm-engine-fragment",
                                                               "drm-engine-compute"};
  struct gpu_info_asahi *gpu_info = container_of(info, struct gpu_info_asahi, base);
  static char *line = NULL;
  static size_t line_buf_size = 0;
  ssize_t count;
  bool driver_matches = false, client_id_set = false, have_resident = false;
  unsigned cid = 0;
  uint64_t engine_ns[asahi_engine_count] = {0}, total = 0, resident = 0;
  bool engine_valid[asahi_engine_count] = {false}, have_total = false;
  nvtop_time current_time;
  nvtop_get_current_time(&current_time);

  while ((count = getline(&line, &line_buf_size, fdinfo_file)) != -1) {
    char *key, *val;
    if (count > 0 && line[count - 1] == '\n')
      line[--count] = '\0';
    if (!extract_drm_fdinfo_key_value(line, &key, &val))
      continue;
    if (!strcmp(key, "drm-driver")) {
      driver_matches = is_asahi_driver(val);
    } else if (!strcmp(key, drm_client_id)) {
      char *end;
      cid = strtoul(val, &end, 10);
      client_id_set = !*end;
    } else if (!strcmp(key, "drm-total-system")) {
      have_total = parse_fdinfo_size(val, &total);
    } else if (!strcmp(key, "drm-resident-system")) {
      have_resident = parse_fdinfo_size(val, &resident);
    } else if (!strncmp(key, "drm-maxfreq-", 12)) {
      unsigned mhz;
      if (parse_fdinfo_frequency_mhz(val, &mhz) && mhz > gpu_info->maxfreq_mhz)
        gpu_info->maxfreq_mhz = mhz;
    } else {
      for (unsigned e = 0; e < asahi_engine_count; ++e) {
        if (strcmp(key, engine_keys[e]))
          continue;
        char *end;
        uint64_t ns = strtoull(val, &end, 10);
        if (end != val && !strcmp(end, " ns")) {
          engine_ns[e] = ns;
          engine_valid[e] = true;
        }
      }
    }
  }
  if (!driver_matches || !client_id_set)
    return false;

  if (have_resident)
    SET_GPUINFO_PROCESS(process_info, gpu_memory_usage, resident);
  else if (have_total)
    SET_GPUINFO_PROCESS(process_info, gpu_memory_usage, total);
  if (engine_valid[asahi_vertex] || engine_valid[asahi_fragment])
    SET_GPUINFO_PROCESS(process_info, gfx_engine_used, engine_ns[asahi_vertex] + engine_ns[asahi_fragment]);
  if (engine_valid[asahi_compute])
    SET_GPUINFO_PROCESS(process_info, compute_engine_used, engine_ns[asahi_compute]);

  struct unique_cache_id ucid = {.client_id = cid, .pid = process_info->pid};
  struct asahi_process_info_cache *cache_entry;
  HASH_FIND_CLIENT(gpu_info->last_update_process_cache, &ucid, cache_entry);
  if (cache_entry) {
    uint64_t elapsed = nvtop_difftime_u64(cache_entry->last_measurement_tstamp, current_time);
    HASH_DEL(gpu_info->last_update_process_cache, cache_entry);
    // The engines overlap (a render's vertex and fragment stages pipeline), so the process keeps the busiest one.
    unsigned usage = 0;
    bool usage_valid = false, graphical = false, compute = false;
    for (unsigned e = 0; e < asahi_engine_count; ++e) {
      if (!engine_valid[e] || !cache_entry->engine_valid[e] || engine_ns[e] < cache_entry->engine_ns[e])
        continue;
      unsigned engine_usage = busy_usage_from_time_usage_round(engine_ns[e], cache_entry->engine_ns[e], elapsed);
      usage_valid = true;
      if (engine_usage > usage)
        usage = engine_usage;
      if (engine_ns[e] != cache_entry->engine_ns[e]) {
        if (e == asahi_compute)
          compute = true;
        else
          graphical = true;
      }
    }
    if (usage_valid)
      SET_GPUINFO_PROCESS(process_info, gpu_usage, usage);
    if (graphical)
      process_info->type |= gpu_process_graphical;
    if (compute)
      process_info->type |= gpu_process_compute;
  } else {
    cache_entry = calloc(1, sizeof(*cache_entry));
    if (!cache_entry)
      return true;
    cache_entry->client_id = ucid;
  }
  memcpy(cache_entry->engine_ns, engine_ns, sizeof(engine_ns));
  memcpy(cache_entry->engine_valid, engine_valid, sizeof(engine_valid));
  cache_entry->last_measurement_tstamp = current_time;
  HASH_ADD_CLIENT(gpu_info->current_update_process_cache, cache_entry);
  return true;
}

static void add_asahi_card(struct nvtop_device *dev, struct list_head *devices, unsigned *count) {
  struct nvtop_device *parent;
  const char *driver, *syspath;
  if (nvtop_device_get_parent(dev, &parent) < 0)
    return;
  if (nvtop_device_get_driver(parent, &driver) < 0 || !is_asahi_driver(driver))
    return;
  if (nvtop_device_get_syspath(parent, &syspath) < 0)
    return;

  struct gpu_info_asahi *gpu = &gpu_infos[asahi_gpu_count++];
  gpu->base.vendor = &gpu_vendor_asahi;
  gpu->card_device = nvtop_device_ref(dev);
  gpu->driver_device = nvtop_device_ref(parent);
  snprintf(gpu->stats_path, sizeof(gpu->stats_path), "%s/agx_stats", syspath);
  snprintf(gpu->memory_path, sizeof(gpu->memory_path), "%s/gpu_memory", syspath);
  list_add_tail(&gpu->base.list, devices);
  processinfo_register_fdinfo_callback(parse_drm_fdinfo_asahi, &gpu->base);
  (*count)++;
}

static bool gpuinfo_asahi_get_device_handles(struct list_head *devices_list, unsigned *count) {
  *count = 0;
  nvtop_device_enumerator *enumerator;
  if (nvtop_enumerator_new(&enumerator) < 0)
    return false;
  if (nvtop_device_enumerator_add_match_subsystem(enumerator, "drm", true) < 0 ||
      nvtop_device_enumerator_add_match_property(enumerator, "DEVNAME", "/dev/dri/*") < 0) {
    nvtop_enumerator_unref(enumerator);
    return false;
  }
  unsigned num_devices = 0;
  for (nvtop_device *device = nvtop_enumerator_get_device_first(enumerator); device;
       device = nvtop_enumerator_get_device_next(enumerator))
    num_devices++;
  gpu_infos = calloc(num_devices ? num_devices : 1, sizeof(*gpu_infos));
  if (!gpu_infos) {
    nvtop_enumerator_unref(enumerator);
    return false;
  }
  for (nvtop_device *device = nvtop_enumerator_get_device_first(enumerator); device;
       device = nvtop_enumerator_get_device_next(enumerator)) {
    const char *devname;
    if (nvtop_device_get_devname(device, &devname) < 0)
      continue;
    if (strstr(devname, "/dev/dri/card"))
      add_asahi_card(device, devices_list, count);
  }
  nvtop_enumerator_unref(enumerator);
  return true;
}

// The SoC from the GPU's device tree compatible string ("apple,agx-t8140").
static void asahi_device_name(struct gpu_info_asahi *gpu, char *name, size_t size) {
  static const struct {
    const char *soc, *marketing;
  } socs[] = {
      {"t8103", "M1"},     {"t6000", "M1 Pro"}, {"t6001", "M1 Max"}, {"t6002", "M1 Ultra"}, {"t8112", "M2"},
      {"t6020", "M2 Pro"}, {"t6021", "M2 Max"}, {"t6022", "M2 Ultra"}, {"t8122", "M3"},     {"t6030", "M3 Pro"},
      {"t6031", "M3 Max"}, {"t8132", "M4"},     {"t6040", "M4 Pro"}, {"t6041", "M4 Max"},   {"t8140", "A18 Pro"},
  };
  const char *syspath;
  char path[PATH_MAX], compatible[128] = {0};
  snprintf(name, size, "Apple GPU");
  if (nvtop_device_get_syspath(gpu->driver_device, &syspath) < 0)
    return;
  snprintf(path, sizeof(path), "%s/of_node/compatible", syspath);
  FILE *f = fopen(path, "r");
  if (!f)
    return;
  size_t n = fread(compatible, 1, sizeof(compatible) - 1, f);
  fclose(f);
  compatible[n] = '\0';
  const char *soc = strstr(compatible, "agx-");
  if (!soc)
    return;
  soc += 4;
  for (size_t i = 0; i < sizeof(socs) / sizeof(*socs); ++i)
    if (!strncmp(soc, socs[i].soc, strlen(socs[i].soc))) {
      snprintf(name, size, "Apple %s GPU", socs[i].marketing);
      return;
    }
  snprintf(name, size, "Apple GPU (%.16s)", soc);
}

static void gpuinfo_asahi_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_asahi *gpu = container_of(_gpu_info, struct gpu_info_asahi, base);
  struct gpuinfo_static_info *static_info = &gpu->base.static_info;
  RESET_ALL(static_info->valid);
  static_info->integrated_graphics = true;
  static_info->encode_decode_shared = false;
  static_info->memory_shared_with_host = true;
  asahi_device_name(gpu, static_info->device_name, sizeof(static_info->device_name));
  SET_VALID(gpuinfo_device_name_valid, static_info->valid);
}

// One "key value" line; *number is set when the value is an unsigned integer (not "unsupported").
static bool read_key(FILE *f, char *key, size_t key_size, uint64_t *number, bool *numeric) {
  char line[256];
  if (!fgets(line, sizeof(line), f))
    return false;
  char *space = strchr(line, ' ');
  *numeric = false;
  if (!space) {
    key[0] = '\0';
    return true;
  }
  *space = '\0';
  snprintf(key, key_size, "%s", line);
  char *end;
  *number = strtoull(space + 1, &end, 10);
  *numeric = end != space + 1 && (*end == '\n' || *end == '\0');
  return true;
}

static void gpuinfo_asahi_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_asahi *gpu = container_of(_gpu_info, struct gpu_info_asahi, base);
  struct gpuinfo_dynamic_info *dynamic_info = &gpu->base.dynamic_info;
  char key[64];
  uint64_t value;
  bool numeric;
  bool busy = false, util = false, util_valid_status = false, achieved = false;
  uint64_t busy_ns = 0, util_device = 0, requested_hz = 0;
  nvtop_time now;

  RESET_ALL(dynamic_info->valid);
  nvtop_get_current_time(&now);

  FILE *f = fopen(gpu->stats_path, "r");
  if (f) {
    while (read_key(f, key, sizeof(key), &value, &numeric)) {
      if (!numeric) {
        continue;
      } else if (!strcmp(key, "busy_ns")) {
        busy = true;
        busy_ns = value;
      } else if (!strcmp(key, "util_device_percent")) {
        util = true;
        util_device = value;
      } else if (!strcmp(key, "achieved_frequency_hz")) {
        achieved = true;
        SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed, (unsigned)(value / 1000000));
      } else if (!strcmp(key, "requested_frequency_hz")) {
        requested_hz = value;
      } else if (!strcmp(key, "power_mw")) {
        SET_GPUINFO_DYNAMIC(dynamic_info, power_draw, (unsigned)value);
      }
    }
    fclose(f);
    // A file without a qualified utilisation line (status not valid) leaves util unset.
    util_valid_status = util;
    // Older GPUs report only the nominal frequency of the state the firmware asked for.
    if (!achieved && requested_hz)
      SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed, (unsigned)(requested_hz / 1000000));
  }
  // Busy time over the refresh interval is an exact mean; the firmware's utilisation is a short window.
  if (busy && gpu->busy_valid && busy_ns >= gpu->busy_ns) {
    uint64_t elapsed = nvtop_difftime_u64(gpu->busy_tstamp, now);
    if (elapsed)
      SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, busy_usage_from_time_usage_round(busy_ns, gpu->busy_ns, elapsed));
  } else if (util_valid_status && util_device <= 100) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, (unsigned)util_device);
  }
  gpu->busy_valid = busy;
  gpu->busy_ns = busy_ns;
  gpu->busy_tstamp = now;
  if (gpu->maxfreq_mhz)
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed_max, gpu->maxfreq_mhz);

  f = fopen(gpu->memory_path, "r");
  if (f) {
    uint64_t limit = 0, used = 0;
    bool have_limit = false, have_used = false;
    while (read_key(f, key, sizeof(key), &value, &numeric)) {
      if (!numeric)
        continue;
      if (!strcmp(key, "limit")) {
        limit = value;
        have_limit = value != 0;
      } else if (!strcmp(key, "user_resident") || !strcmp(key, "kernel_resident")) {
        used += value;
        have_used = true;
      }
    }
    fclose(f);
    if (have_limit) {
      SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, limit);
      if (have_used) {
        SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, used);
        SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, used < limit ? limit - used : 0);
        SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, (unsigned)(used * 100 / limit));
      }
    }
  }
}

static void gpuinfo_asahi_get_running_processes(struct gpu_info *_gpu_info) {
  // The fdinfo callback fills the processes; swap the per-client caches for the next update.
  struct gpu_info_asahi *gpu = container_of(_gpu_info, struct gpu_info_asahi, base);
  struct asahi_process_info_cache *entry, *tmp;
  HASH_ITER(hh, gpu->last_update_process_cache, entry, tmp) {
    HASH_DEL(gpu->last_update_process_cache, entry);
    free(entry);
  }
  gpu->last_update_process_cache = gpu->current_update_process_cache;
  gpu->current_update_process_cache = NULL;
}
