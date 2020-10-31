/*
  Simple single layer neural network
  
  Input layer
  Hidden layer
  Output layer

  Compiling
  ---------
  
    $ make
*/

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//
double sigmoid(double x)
{ return 1 / (1 + exp(-x)); }

//
double d_sigmoid(double x)
{ return x * (1 - x); }

//
void test(double w1[11][11],
	  double w2[11],
	  unsigned long long n1,
	  unsigned long long n2)
{
  //
  double s = 0.0, l[11], I[11];

  //Read test input values
  printf("Enter test values:\n");

  for (unsigned long long i = 1; i <= n1; i++)
    {
      printf("Enter input %llu: ", i);
      scanf("%lf", &I[i]);
    }

  //Neural network result calculation
  for (unsigned long long i = 1; i <= n2; i++)
    {
      s = 0.0;
      
      for (unsigned long long j = 1; j <= n1; j++)
	s += I[j] * w1[j][i];
      
      l[i] = sigmoid(s);
    }

  //
  s = 0.0;
  
  for (unsigned long long i = 1; i <= n2; i++)
    s += l[i] * w2[i];
  
  //
  printf("Output: %0.15lf\n", sigmoid(s));
}

//
int main(int argc, char **argv)
{
  //
  unsigned long long n = 0;
  
  //
  unsigned long long n1 = 0; //Neurons in input layer
  double w1[11][11];     //Weights of input layer
  
  //
  unsigned long long n2 = 0; //Neurons in hidden layer
  double w2[11];         //Weights of the hidden layer

  //Input
  double I[101][11];
  
  //Output
  double O[101];

  //
  double l1[101][11];    //Output of the input layer
  double l1_d[101][11];  //Derivative (to be back propagated)
  
  //
  double l2[101];        //Output of the hidden layer
  double l2_d[101];      //Derivative (to be back propagated)
  
  //
  double s = 0.0, err = 0.0, alpha = 1.0;

  //
  char c = 0;
  
  //
  srand(time(NULL));
  
  //
  /* printf("Enter the number of neurons in the input layer: "); */
  /* scanf("%llu", &n1); */

  /* // */
  /* printf("Enter the number of neurons in the hidden layer: "); */
  /* scanf("%llu", &n2); */
  
  /* //(in1, in2, in3, out) */
  /* printf("Enter the number of training entries: "); */
  /* scanf("%llu", &n); */
  
  //Parameters
  n1 = 3;
  n2 = 5;
  n  = 8;
  
  /* // */
  /* for (unsigned long long i = 0; i < n; i++) */
  /*   { */
  /*     printf("Entry %llu:\n", i); */
  /*     for (unsigned long long j = 1; j <= n1; j++) */
  /* 	{ */
  /* 	  printf("Enter I[%llu][%llu]: ", i, j); */
  /* 	  scanf("%lf", &I[i][j]); */
  /* 	} */
  
  /*     printf("Enter expected output value O[%llu]: ", i); */
  /*     scanf("%lf", &O[i]); */
  
  /*     putchar('\n'); */
  /*   } */
  
  //XOR tables
  I[0][1] = 0;
  I[0][2] = 0;
  I[0][3] = 0;
  O[0]    = 0;
  
  I[1][1] = 1;
  I[1][2] = 1;
  I[1][3] = 1;
  O[1]    = 1;
  
  I[2][1] = 0;
  I[2][2] = 0;
  I[2][3] = 1;
  O[2]    = 1;
  
  I[3][1] = 1;
  I[3][2] = 1;
  I[3][3] = 0;
  O[3]    = 0;
  
  I[4][1] = 0;
  I[4][2] = 1;
  I[4][3] = 0;
  O[4]    = 1;
  
  I[5][1] = 1;
  I[5][2] = 0;
  I[5][3] = 1;
  O[5]    = 0;

  I[6][1] = 1;
  I[6][2] = 0;
  I[6][3] = 0;
  O[6]    = 1;

  I[7][1] = 0;
  I[7][2] = 1;
  I[7][3] = 1;
  O[7]    = 0;

 lbl1:
  //Move in cols
  for (unsigned long long i = 1; i <= n2; i++)
    {
      //
      for (unsigned long long j = 1; j <= n1; j++)
	w1[j][i] = (double)rand() / (double)RAND_MAX;
      
      //
      w2[i] = (double)rand() / (double)RAND_MAX;
    }
  
 lbl2:
  //The real stuff happen here
  for (unsigned long long i = 0;; i++) //Infinite loop
    {
      //For each entry
      for (unsigned long long j = 1; j <= n; j++)
	{
	  //Building l1
	  for (unsigned long long k = 1; k <= n2; k++)
	    {
	      s = 0.0;
	      
	      //Dotprod/FMA/MAC
	      for (unsigned long long l = 1; l <= n1; l++)
		s += I[j][l] * w1[l][k];
	      
	      //Update neuron value
	      l1[j][k] = sigmoid(s);
	    }
	}
      
      //
      err = 0.0;
      
      //For each entry
      for (unsigned long long j = 1; j <= n; j++)
	{
	  s = 0.0;
	  
	  //Builsing l2
	  for (unsigned long long k = 1; k <= n2; k++)
	    s += l1[j][k] * w2[k];

	  //
	  l2[j] = sigmoid(s);
	  
	  //Mean absolute error
	  err += fabs(l2[j] - O[j]);
	  
	  //L2 deltas
	  l2_d[j] = (l2[j] - O[j]) * d_sigmoid(l2[j]);
	}
      
      //
      if (!((i + 1) % 1000000))
	{
	  printf("round: %llu; err: %lf\n", i + 1, err);
	}
      
      //
      for (unsigned long long j = 1; j <= n; j++)
	{
	  double _c_ = l2_d[j];
	  
	  for (unsigned long long k = 1; k <= n2; k++)
	    l1_d[j][k] = _c_ * w2[k] * d_sigmoid(l1[j][k]);
	}
      
      //Backprop for w1
      for (unsigned long long j = 1; j <= n1; j++)
	{
	  for (unsigned long long k = 1; k <= n2; k++)
	    {
	      s = 0.0;
	      
	      //
	      for (unsigned long long l = 1; l <= n; l++)
		s += I[l][j] * l1_d[l][k];
	      
	      //
	      w1[j][k] -= alpha * s;
	    }
	}
      
      //
      for (unsigned long long j = 1; j <= n2; j++)
	{
	  s = 0.0;
	  
	  for (unsigned long long k = 1; k <= n; k++)
	    s += l1[k][j] * l2_d[k];
	  
	  //
	  w2[j] -= alpha * s; 
	}
      
      //Menu
      if (err <= 1.0e-3)
	{
	  printf("0 - Test\n");
	  printf("1 - Retrain\n");
	  printf("2 - Continue training\n");
	  printf("3 - Exit\n");
	  
	  c = getchar();
	  getchar();
	  
	  switch (c)
	    {
	      //Testing the neural network
	    case '0':
	      test(w1, w2, n1, n2);
	      break;

	      //Retrain the network 
	    case '1':
	      goto lbl1;

	      //Continue training 
	    case '2':
	      goto lbl2;

	      //Exit
	    case '3':
	      exit(0);
	    }
	}
    }
  
  //
  return 0;
}
