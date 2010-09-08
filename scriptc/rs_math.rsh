#ifndef __RS_MATH_RSH__
#define __RS_MATH_RSH__

// Debugging, print to the LOG a description string and a value.
extern void __attribute__((overloadable))
    rsDebug(const char *, float);
extern void __attribute__((overloadable))
    rsDebug(const char *, float, float);
extern void __attribute__((overloadable))
    rsDebug(const char *, float, float, float);
extern void __attribute__((overloadable))
    rsDebug(const char *, float, float, float, float);
extern void __attribute__((overloadable))
    rsDebug(const char *, const rs_matrix4x4 *);
extern void __attribute__((overloadable))
    rsDebug(const char *, const rs_matrix3x3 *);
extern void __attribute__((overloadable))
    rsDebug(const char *, const rs_matrix2x2 *);
extern void __attribute__((overloadable))
    rsDebug(const char *, int);
extern void __attribute__((overloadable))
    rsDebug(const char *, uint);
extern void __attribute__((overloadable))
    rsDebug(const char *, const void *);
#define RS_DEBUG(a) rsDebug(#a, a)
#define RS_DEBUG_MARKER rsDebug(__FILE__, __LINE__)


#include "rs_cl.rsh"
#include "rs_core.rsh"

extern void __attribute__((overloadable))
    rsSetObject(rs_element *dst, rs_element src);
extern void __attribute__((overloadable))
    rsSetObject(rs_type *dst, rs_type src);
extern void __attribute__((overloadable))
    rsSetObject(rs_allocation *dst, rs_allocation src);
extern void __attribute__((overloadable))
    rsSetObject(rs_sampler *dst, rs_sampler src);
extern void __attribute__((overloadable))
    rsSetObject(rs_script *dst, rs_script src);
extern void __attribute__((overloadable))
    rsSetObject(rs_mesh *dst, rs_mesh src);
extern void __attribute__((overloadable))
    rsSetObject(rs_program_fragment *dst, rs_program_fragment src);
extern void __attribute__((overloadable))
    rsSetObject(rs_program_vertex *dst, rs_program_vertex src);
extern void __attribute__((overloadable))
    rsSetObject(rs_program_raster *dst, rs_program_raster src);
extern void __attribute__((overloadable))
    rsSetObject(rs_program_store *dst, rs_program_store src);
extern void __attribute__((overloadable))
    rsSetObject(rs_font *dst, rs_font src);

extern void __attribute__((overloadable))
    rsClearObject(rs_element *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_type *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_allocation *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_sampler *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_script *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_mesh *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_program_fragment *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_program_vertex *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_program_raster *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_program_store *dst);
extern void __attribute__((overloadable))
    rsClearObject(rs_font *dst);

extern bool __attribute__((overloadable))
    rsIsObject(rs_element);
extern bool __attribute__((overloadable))
    rsIsObject(rs_type);
extern bool __attribute__((overloadable))
    rsIsObject(rs_allocation);
extern bool __attribute__((overloadable))
    rsIsObject(rs_sampler);
extern bool __attribute__((overloadable))
    rsIsObject(rs_script);
extern bool __attribute__((overloadable))
    rsIsObject(rs_mesh);
extern bool __attribute__((overloadable))
    rsIsObject(rs_program_fragment);
extern bool __attribute__((overloadable))
    rsIsObject(rs_program_vertex);
extern bool __attribute__((overloadable))
    rsIsObject(rs_program_raster);
extern bool __attribute__((overloadable))
    rsIsObject(rs_program_store);
extern bool __attribute__((overloadable))
    rsIsObject(rs_font);



// Allocations

// Return the rs_allocation associated with a bound data
// pointer.
extern rs_allocation __attribute__((overloadable))
    rsGetAllocation(const void *);

// Mark the allocation dirty and notify those using it
extern void __attribute__((overloadable))
    rsAllocationMarkDirty(rs_allocation);

// Return the dimensions associated with an allocation.
extern uint32_t __attribute__((overloadable))
    rsAllocationGetDimX(rs_allocation);
extern uint32_t __attribute__((overloadable))
    rsAllocationGetDimY(rs_allocation);
extern uint32_t __attribute__((overloadable))
    rsAllocationGetDimZ(rs_allocation);
extern uint32_t __attribute__((overloadable))
    rsAllocationGetDimLOD(rs_allocation);
extern uint32_t __attribute__((overloadable))
    rsAllocationGetDimFaces(rs_allocation);

// Extract a single element from an allocation.
extern const void * __attribute__((overloadable))
    rsGetElementAt(rs_allocation, uint32_t x);
extern const void * __attribute__((overloadable))
    rsGetElementAt(rs_allocation, uint32_t x, uint32_t y);
extern const void * __attribute__((overloadable))
    rsGetElementAt(rs_allocation, uint32_t x, uint32_t y, uint32_t z);

// Return a random value between 0 (or min_value) and max_malue.
extern int __attribute__((overloadable))
    rsRand(int max_value);
extern int __attribute__((overloadable))
    rsRand(int min_value, int max_value);
extern float __attribute__((overloadable))
    rsRand(float max_value);
extern float __attribute__((overloadable))
    rsRand(float min_value, float max_value);

// return the fractional part of a float
// min(v - ((int)floor(v)), 0x1.fffffep-1f);
extern float __attribute__((overloadable))
    rsFrac(float);

// time
extern int32_t __attribute__((overloadable))
    rsSecond(void);
extern int32_t __attribute__((overloadable))
    rsMinute(void);
extern int32_t __attribute__((overloadable))
    rsHour(void);
extern int32_t __attribute__((overloadable))
    rsDay(void);
extern int32_t __attribute__((overloadable))
    rsMonth(void);
extern int32_t __attribute__((overloadable))
    rsYear(void);

// Return the current system clock in milliseconds
extern int64_t __attribute__((overloadable))
    rsUptimeMillis(void);

// Return the current system clock in nanoseconds
extern int64_t __attribute__((overloadable))
    rsUptimeNanos(void);

// Return the time in seconds since function was last called in this script.
extern float __attribute__((overloadable))
    rsGetDt(void);

// Send a message back to the client.  Will not block and returns true
// if the message was sendable and false if the fifo was full.
// A message ID is required.  Data payload is optional.
extern bool __attribute__((overloadable))
    rsSendToClient(int cmdID);
extern bool __attribute__((overloadable))
    rsSendToClient(int cmdID, const void *data, uint len);

// Send a message back to the client, blocking until the message is queued.
// A message ID is required.  Data payload is optional.
extern void __attribute__((overloadable))
    rsSendToClientBlocking(int cmdID);
extern void __attribute__((overloadable))
    rsSendToClientBlocking(int cmdID, const void *data, uint len);


// Script to Script
enum rs_for_each_strategy {
    RS_FOR_EACH_STRATEGY_SERIAL,
    RS_FOR_EACH_STRATEGY_DONT_CARE,
    RS_FOR_EACH_STRATEGY_DST_LINEAR,
    RS_FOR_EACH_STRATEGY_TILE_SMALL,
    RS_FOR_EACH_STRATEGY_TILE_MEDIUM,
    RS_FOR_EACH_STRATEGY_TILE_LARGE
};

typedef struct rs_script_call {
    enum rs_for_each_strategy strategy;
    uint32_t xStart;
    uint32_t xEnd;
    uint32_t yStart;
    uint32_t yEnd;
    uint32_t zStart;
    uint32_t zEnd;
    uint32_t arrayStart;
    uint32_t arrayEnd;
} rs_script_call_t;

extern void __attribute__((overloadable))
    rsForEach(rs_script script, rs_allocation input,
              rs_allocation output, const void * usrData);

extern void __attribute__((overloadable))
    rsForEach(rs_script script, rs_allocation input,
              rs_allocation output, const void * usrData,
              const rs_script_call_t *);

#endif