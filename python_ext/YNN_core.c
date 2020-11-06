//
#include <math.h>

//
#define ALIGN 64

//
float sigmoid(float x)
{ return 1 / (1 + exp(-x)); }

//
float d_sigmoid(float x)
{ return x * (1 - x); }

//
float reduc_f32(float *a, unsigned long long n)
{
  float r = 0.0;

  for (unsigned long long i = 0; i < n; i++)
    r += a[i];

  return r;
}

//1 cache line
float reduc_f32_AVX(float *a, unsigned long long n)
{
  //
  unsigned long long _n_ = n - (n & 15);
  float r[8] __attribute__((aligned(ALIGN)));
  unsigned long long s = sizeof(float) * _n_;
  
  //
  __asm__ volatile(
		   //Zero up the loop counter
		   "xor %%rcx, %%rcx;\n"

		   //Zero up the partial accumulation registers
		   "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		   "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"

		   //Loop entry
		   "1:;\n"

		   //Load and accumulate
		   "vaddps   (%[_a_], %%rcx), %%ymm0, %%ymm0;\n"
		   "vaddps 32(%[_a_], %%rcx), %%ymm1, %%ymm1;\n"

		   //Next 
		   "add $64, %%rcx;\n"
		   "cmp %[_s_], %%rcx;\n"
		   "jl 1b;\n"
		   
		   //Sum up the partial accumulation register
		   "vaddps %%ymm1, %%ymm0, %%ymm0;\n"
		   
		   //Store result
		   "vmovaps %%ymm0, (%[_r_]);\n"
		   
		   : //output
		   : //input
		     [_a_] "r" (a),
		     [_s_] "r" (s),
		     [_r_] "r" (r)
		   : //clobber
		     "cc", "memory", "rcx",
		     "ymm0", "ymm1"
		   );

  //
  for (unsigned long long i = _n_; i < n; i++)
    r[0] += a[i];
  
  //
  return (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
}

//
float dotprod_f32(float *a, float *b, unsigned long long n)
{
  float r = 0.0;

  for (unsigned long long i = 0; i < n; i++)
    r += a[i] * b[i];

  return r;
}

//1 cache line
float dotprod_f32_AVX(float *a,
		      float *b,
		      unsigned long long n)
{
  //
  unsigned long long _n_ = n - (n & 15);
  float r[8] __attribute__((aligned(ALIGN)));
  unsigned long long s = sizeof(float) * _n_;
  
  //
  __asm__ volatile(
		   "xor %%rcx, %%rcx;\n"
		   
		   "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		   "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"
		   
		   "1:;\n"
		   
		   "vmovups   (%[_a_], %%rcx), %%ymm2;\n"
		   "vmovups 32(%[_a_], %%rcx), %%ymm3;\n"
		   
		   "vfmadd231ps   (%[_b_], %%rcx), %%ymm2, %%ymm0;\n"
		   "vfmadd231ps 32(%[_b_], %%rcx), %%ymm3, %%ymm1;\n"
		   
		   "add $64, %%rcx;\n"
		   "cmp %[_s_], %%rcx;\n"
		   "jl 1b;\n"
		   
		   "vaddps %%ymm1, %%ymm0, %%ymm0;\n"
		   
		   "vmovaps %%ymm0, (%[_r_]);\n"
		   
		   : //output
		   : //input
		     [_a_] "r" (a),
		     [_b_] "r" (b),
		     [_s_] "r" (s),
		     [_r_] "r" (r)
		   : //clobber
		     "cc", "memory", "rcx",
		     "ymm0", "ymm1", "ymm2", "ymm3"
		   );

  for (unsigned long long i = _n_; i < n; i++)
    r[0] += a[i];
  
  //
  return (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
}
