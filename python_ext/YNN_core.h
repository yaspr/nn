#ifndef YNN_CORE_H
#define YNN_CORE_H

//
float sigmoid(float x);
float d_sigmoid(float x);
float reduc_f32(float *a, unsigned long long n);
float reduc_f32_AVX(float *a, unsigned long long n);
float dotprod_f32(float *a, float *b, unsigned long long n);
float dotprod_f32_AVX(float *a, float *b, unsigned long long n);

#endif
