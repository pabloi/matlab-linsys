#include <stdlib.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"
#include <gsl/gsl_linalg.h>
#include "utils.h"

/*Global function*/
#define zeros(X,Y) mxCreateDoubleMatrix((X),(Y),mxREAL)

void mexFunction(
				 int nlhs, mxArray *plhs[], /*outputs*/
				 int nrhs, const mxArray *prhs[] /*inputs*/
				 )
{
	const mxArray *A; /*square matrix*/
  mxArray *L,*P,*E;
  double *prA=mxGetPr(A);
  double *prL, *prE, *prP;
	char	 buffer[100];
  int Nx=mxGetM(A);

  /************* Check for proper number of arguments ********/
  /* Inputs:   [A]
  /* Outputs:  [L,p,E]  L=chol decomp, p=permutation, E=perturbation matrix*/
  if( nrhs!=1){
    sprintf(buffer,
      "mchol: args = square, hermitian matrix A");
    mexErrMsgTxt(buffer);
  }

  /* Check for argument sizes: TO DO */

  /************* Create Output Matrices ********/
  L=zeros(Nx,Nx);
  P=mxCreateNumericArray(1, &Nx, mxINT16_CLASS, mxREAL);
  P=zeros(Nx,1);
  E=zeros(Nx,1);
  prL=mxGetPr(L);
  prP=mxGetPr(P);
  prE=mxGetPr(E);

  /************* Do Computation and Return ********/
  /*First, create matrices for function*/
  gsl_permutation *p= gsl_permutation_alloc(Nx);
  gsl_vector *v=gsl_vector_calloc(Nx);
  gsl_matrix *mxx= gsl_matrix_calloc(Nx,Nx);

  /*Second, allocate the data to the matrix to factorize*/
  double *txx=mxx->data;
  subcopy(txx,1, prL,1, Nx*Nx); /*Copy Nx*Nx elements */
  subcopy(p->data,1,prP,1, Nx);
  double *tv =v->data;
  subcopy(tv,1,prE,1, Nx);

  /*Third, do the computation. Requires GNU scientific library*/
  int boo=gsl_linalg_mcholesky_decomp (mxx,p,v);

  /*Fourth, populate the global mxArrays L,P,E*/
  subcopy(prL,1,txx,1, Nx*Nx); /*Copy Nx*Nx elements */
  /* subcopy(prP,1,(double) p->data,1, Nx); */
  subcopy(prE,1,tv,1, Nx);

  int i=0;
  plhs[i++]=L;

  if(nlhs>i) plhs[i++]=P;
  else mxDestroyArray(P);

  if(nlhs>i) plhs[i++]=E;
  else mxDestroyArray(E);
  }
