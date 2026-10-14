/**
 * AVCS (Artificial Visual Cortex System) High-Throughput C/C++ IPC Header.
 *
 * Provides C99 / C++17 compatible memory-mapped struct definitions and
 * sub-microsecond state reading for native performance applications.
 */

#ifndef AVCS_IPC_H
#define AVCS_IPC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AVCS_MAGIC 0x53435641  /* "AVCS" in Little-Endian */
#define AVCS_VERSION 1
#define AVCS_DEFAULT_MMAP_NAME "avcs_perception_mmap"

#pragma pack(push, 1)

typedef struct {
    uint8_t  magic[4];       /* "AVCS" */
    uint32_t version;        /* IPC protocol version (1) */
    uint32_t flags;          /* Reserved flags */
    uint64_t seq_num;        /* Monotonically increasing sequence number */
    double   timestamp;      /* Unix epoch monotonic timestamp */
    uint32_t payload_len;    /* Byte length of JSON / binary payload */
    uint32_t crc32;          /* CRC32 checksum of payload */
} avcs_ipc_header_t;

typedef struct {
    float x;
    float y;
} avcs_point_t;

typedef struct {
    uint64_t     sequence_number;
    double       timestamp;
    float        latency_ms;
    float        fps;
    avcs_point_t gaze_fixation;
    float        surprise_score;
    char         primary_category[32];
    char         stability[16];
    uint32_t     affordances_count;
} avcs_compact_state_t;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* AVCS_IPC_H */

