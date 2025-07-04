#if defined(__x86_64__) && !defined(ISPC)
  typedef _Float16 __fp16;
#endif
#ifdef ISPC
  typedef float16 data_t;
#else
  typedef __fp16 data_t;
#endif
// #if defined(ISPC_TARGET_AVX2) || defined(__AVX2__)
//   typedef float itmd_t;
// #else
//   typedef data_t itmd_t;
// #endif
typedef float itmd_t;
typedef float otpt_t;

#define GEMM cblas_hgemm
#define HEAD_DIM 128 // Constant for all models
#define BLOCK_SIZE 16

#if defined(LLAMA3_8B)
  #define NUM_LAYERS 32
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(LLAMA2_7B)
  #define NUM_LAYERS 32
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (32 / TP_DEGREE)
#elif defined(LLAMA2_13B)  
  #define NUM_LAYERS 40
  #define NUM_Q_HEADS (40 / TP_DEGREE)
  #define NUM_KV_HEADS (40 / TP_DEGREE)
#elif defined(LLAMA2_70B) || defined(LLAMA3_70B)
  #define NUM_LAYERS 80
  #define NUM_Q_HEADS (64 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(YI_9B) 
  #define NUM_LAYERS 48
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (4 / TP_DEGREE)
#elif defined(YI_34B) 
  #define NUM_LAYERS 60
  #define NUM_Q_HEADS (56 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#else
  #error "Please define the model"
#endif

#define QH_PER_KVH (NUM_Q_HEADS / NUM_KV_HEADS)
#define BLOCK_NELEM (NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM)

#define MAX_BATCH_SIZE 4096
#define MAX_WS 256
#define MAX_TOK_NUM 1048576 // Maxinum number of token's KV to be scanned in one iteration
#define MAX_TASK_NUM (MAX_BATCH_SIZE + MAX_WS)