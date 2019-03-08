/*  utils.c - 
    Copyright (C) 

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

/*** utils.c ***/

#include <stdlib.h>
#include <math.h>

#include "mex.h"
#include "matrix.h"
#include "utils.h"

/* Computes matrix product C= A*B where A is a (LxM)-matrix and B is
 * (MxN)-matrix
 */
void multm(const double *prA, const double *prB, double *prC, 
        int L, int M, int N)
{
    int l,m,n;
    for(l=0; l<L; l++)
        for(n=0; n<N; n++){
            prC[l+L*n] = 0.0;
            for(m=0; m<M; m++)
                prC[l+L*n] += prA[l+L*m]*prB[m+M*n];
        }
}

/* Computes matrix product C= A*B' where A is (LxM) and B is (NxM)
*/
void multmtr(const double *prA, const double *prB, double *prC, 
        int L, int M, int N)
{
    int l,m,n;
    for(l=0; l<L; l++)
        for(n=0; n<N; n++){
            prC[l+L*n] = 0.0;
            for(m=0; m<M; m++)
                prC[l+L*n] += prA[l+L*m]*prB[n+N*m];
        }
}

/* Computes matrix product R= A(k,:,:)*B(j,:,:) 
 * where A(k,:,:) is a (LxM)-matrix and B(j,:,:) is a (MxN)-matrix
 */
void submult3d(const double *A, int k, const double *B, int j, 
        double *R, int L, int M, int N)
{
    A +=(k-1)*L*M;
    B +=(j-1)*M*N;
    multm(A, B, R, L, M, N);
}

/* Computes matrix product R= A(k,:,:)*B(j,:,:)'
 * where A(k,:,:) is a (LxM)-matrix and B(j,:,:) is a (NxM)-matrix.
 * R is (NxL)
 */
void submult3dtr(const double *A, int k, const double *B, int j, 
        double *R, int L, int M, int N)
{
    A +=(k-1)*L*M;
    B +=(j-1)*M*N;
    multmtr(A, B, R, L, M, N);
}

/*******************************
 * outerSum
 *
 * Computes sum{t=t1:t2,s=s1:s2}{A(:,t)*B(:,s)'} where A and B are 
 *(MxT) and(NxT), respectively.
 */
void outerSum(const double *prA, int M, int t1, int t2,
        const double *prB, int N, int s1, int s2, 
        double *prR)
{
    int t,m,n;
    prA += M*(t1-1);
    prB += N*(s1-1);
    for(t=t1; t<=t2; t++){
        for(m=0; m<M; m++)
            for(n=0; n<N; n++)
                prR[m+M*n] += prA[m]*prB[n];
        prA+=M; prB+=N;
    }	
}

void eyem(double *pR, int r)
{
    int i;
    zerosm(pR,r,r);
    for(i=0;i<r;i++) pR[i+r*i]=1.0;
}
void zerosm(double *pR, int N, int M)
{
    int n,m;
    for(n=0; n<N; n++)
        for(m=0; m<M; m++)
            pR[n+N*m]= 0.0;
}

/*  Changes up to HERE */

/*******************************
 * Copy the D-dimensional matrix 
 * B(:,...:,j) of total size n 
 * into the kth "position" of 
 * (D+1)-dimensional matrix A.
 * A(:,...,:,k) = B(:,...:,j);
 *
 * No Dimension Checking!!!!
 */
void subcopy(double *pA, int k, const double *pB, int j, int n)
{
    int i;
    pA += (k-1)*n;
    pB += (j-1)*n;
    for (i=0;i<n;i++) pA[i]=pB[i];
}

/*******************************
 * Increment the  the kth "position" of 
 * (D+1)-dimensional matrix A by
 * the D-dimensional matrix B(:,...:,j)
 * of total size n (n=prod(size(B)))
 * A(:,...,:,k) = A(:,...,:,k) + B(:,...:,j);
 *
 * No Dimension Checking!!!!
 */
void subinc(double *pA, int k, const double *pB, int j, int n)
{
    int i;
    pA += (k-1)*n;
    pB += (j-1)*n;
    for (i=0;i<n;i++) pA[i]+=pB[i];
}

/*******************************
 * Decrement the  the kth "position" of 
 * (D+1)-dimensional matrix A by
 * the D-dimensional matrix B(:,...:,j)
 * of total size n (n=prod(size(B)))
 * A(:,...,:,k) = A(:,...,:,k) - B(:,...:,j);
 *
 * No Dimension Checking!!!!
 */
void subdec(double *pA, int k, const double *pB, int j, int n)
{
    int i;
    pA += (k-1)*n;
    pB += (j-1)*n;
    for (i=0;i<n;i++) pA[i]-=pB[i];
}

/*******************************
 * submult
 *
 * Computes A*B where A and B are arbitrary matrices.
 % (LxM) and (MxN)
 * It creates a new matrix for storing the result.
 * !! You are responsible for freeing it. !!
 */
mxArray * submult (const mxArray *A, int k, 
        const mxArray *B, int j, 
        int L, int M, int N)
{
    int l,m,n;
    mxArray *R;
    double *prA,*prB,*prR;

    R=mxCreateDoubleMatrix(L,N,mxREAL);
    /*Note: R already contains zeros*/

    prA = mxGetPr(A);
    prB = mxGetPr(B);
    prR = mxGetPr(R);

    prA += (k-1)*L*M;
    prB += (j-1)*M*N;

    for (l=0; l<L; l++)
        for(n=0; n<N; n++){
            prR[l+L*n] = 0.0;
            for(m=0; m<M; m++)
                prR[l+L*n] += prA[l+L*m]*prB[m+M*n];
        }

    return R;	
}

/* mxInnerProd
 * 
 * Computes A'*B where A and B are vectors.
 */
double innerProd (const double *prA, const double *prB, int N)
{
    int idx;
    double ret= 0.0;
    for(idx=0; idx<N;idx++)
        ret += prA[idx]*prB[idx];
    return ret;
}


