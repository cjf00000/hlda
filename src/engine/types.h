#ifndef __TYPES
#define __TYPES

typedef unsigned int TTopic;
typedef unsigned int TWord;
typedef unsigned int TDoc;
typedef unsigned int TLen;
typedef unsigned int TCount;
typedef unsigned int TIndex;
typedef unsigned int TCoord;
typedef unsigned int TIter;
typedef int TId;
typedef long long TSize;
typedef float TProb;
typedef double TLikehood;
typedef enum {
    row_partition,
    column_partition
} PartitionType;
// monolith require a preprocess of the corpus.
// If you don't want to, use separate instead.
typedef enum {
    monolith,
    separate
} LocalMergeStyle;
typedef struct {
    TCoord r, c;
} Entry;
typedef struct {
    TCoord k;
    TCount v;
} SpEntry;

#define ALIGN_SIZE 32
#define MAX_NUM_TOPICS 100000

#endif
