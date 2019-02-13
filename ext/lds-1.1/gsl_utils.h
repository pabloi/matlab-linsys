/*  gsl_utils.h - header file for linear algebra routines
    Copyright (C) 2005 Sen Cheng

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

/*  
    Almost all the code in this file was taken from various files of the GNU
    scientific library version 1.6.
*/

#ifndef GSL_UTILS_H
#define GSL_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif


/* Code taken from GNU scientific library (gsl 1.6), begin. */

struct gsl_block_struct
{
  size_t size;
  double *data;
};
typedef struct gsl_block_struct gsl_block;
                                                                                
gsl_block *gsl_block_alloc (const size_t n);
gsl_block *gsl_block_calloc (const size_t n);
void gsl_block_free (gsl_block * b);

typedef struct
{
  size_t size;
  size_t stride;
  double *data;   
  gsl_block *block;
  int owner;
}
gsl_vector;
                                                                                
typedef struct
{
  gsl_vector vector;
} _gsl_vector_view;
                                                                                
typedef _gsl_vector_view gsl_vector_view;


typedef struct
{
  size_t size1;
  size_t size2;
  size_t tda;
  double * data;
  gsl_block * block;
  int owner;
} gsl_matrix;
                                                                                
typedef struct
{
  gsl_matrix matrix;
} _gsl_matrix_view;
                                                                                
typedef _gsl_matrix_view gsl_matrix_view;

struct gsl_permutation_struct
{
  size_t size;
  size_t *data;
};
                                                                                
typedef struct gsl_permutation_struct gsl_permutation;

int gsl_linalg_LU_svx (const gsl_matrix * LU,
                       const gsl_permutation * p,
                       gsl_vector * x);
                                                                                
                                                                                
gsl_permutation *gsl_permutation_alloc (const size_t n);
void gsl_permutation_init (gsl_permutation * p);
void gsl_permutation_free (gsl_permutation * p);
int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum);

int gsl_linalg_LU_invert (const gsl_matrix * LU,
                          const gsl_permutation * p,
                          gsl_matrix * inverse);
                                                                                
double gsl_linalg_LU_det (gsl_matrix * LU, int signum);


void gsl_permutation_init (gsl_permutation * p);
int gsl_permutation_swap (gsl_permutation * p, const size_t i, const size_t j);

int gsl_matrix_swap_rows(gsl_matrix * m, const size_t i, const size_t j);
double   gsl_matrix_get(const gsl_matrix * m, const size_t i, const size_t j);
void    gsl_matrix_set(gsl_matrix * m, const size_t i, const size_t j, const double x);
void gsl_matrix_set_identity (gsl_matrix * m);
_gsl_vector_view gsl_matrix_column (gsl_matrix * m, const size_t j);

                                                                                
gsl_matrix* gsl_matrix_alloc (const size_t n1, const size_t n2);
gsl_matrix* gsl_matrix_calloc (const size_t n1, const size_t n2);
void gsl_matrix_free (gsl_matrix * m);
_gsl_matrix_view gsl_matrix_view_array (double * base,
        const size_t n1, const size_t n2);



int gsl_permute_vector (const gsl_permutation * p, gsl_vector * v);
int gsl_permute_vector_inverse (const gsl_permutation * p, gsl_vector * v);

/*
 *  * Enumerated and derived types
 *   */
#define CBLAS_INDEX size_t  /* this may vary between platforms */

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
typedef  enum CBLAS_UPLO        CBLAS_UPLO_t;
typedef  enum CBLAS_TRANSPOSE   CBLAS_TRANSPOSE_t;
typedef  enum CBLAS_DIAG        CBLAS_DIAG_t;
#define INDEX int


int  gsl_blas_dtrsv (CBLAS_UPLO_t Uplo,
        CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag,
        const gsl_matrix * A,
        gsl_vector * X);


void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
        const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
        const int N, const double *A, const int lda, double *X,
        const int incX);


/* Code taken from GNU scientific library, end. */


/* new functions */
void myinv(const size_t n, gsl_matrix *m, gsl_matrix *inv, 
        gsl_permutation * p, double *det);


#ifdef __cplusplus
}
#endif

#endif /* GSL_UTILS_H */
