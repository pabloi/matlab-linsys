/*  gsl_utils.c - linear algebra routines
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

#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include "gsl_utils.h"

void myinv(const size_t n, gsl_matrix *m, gsl_matrix *inv, 
        gsl_permutation * p, double *det)
{
    int signum;
    gsl_linalg_LU_decomp(m, p, &signum);
    gsl_linalg_LU_invert(m,p,inv);
    *det= gsl_linalg_LU_det(m,signum);
}

void warning(const char error_text[])
{
    mexWarnMsgTxt(error_text);
}

void error(char error_text[])
{
    fprintf(stderr,"Fatal error in gsl_utils.c: %s\n", error_text);
    exit(1);
}

#define GSL_SUCCESS 0
#define REAL double
#define BASE double
#define ATOMIC double
#define MULTIPLICITY 1
#define IN_FORMAT "%lg"
#define OUT_FORMAT "%g"
#define ATOMIC_IO ATOMIC
#define ZERO 0.0
#define ONE 1.0

#define NULL_VECTOR_VIEW {{0, 0, 0, 0, 0}}
#define NULL_VECTOR {0, 0, 0, 0, 0}
#define NULL_MATRIX {0, 0, 0, 0, 0, 0}
#define NULL_MATRIX_VIEW {{0, 0, 0, 0, 0, 0}}

#define gsl_check_range 1


/* Code taken from GNU scientific library (gsl 1.6), begin. */

#define OFFSET(N, incX) ((incX) > 0 ?  0 : ((N) - 1) * (-(incX)))
#define INT(X) ((int)(X))

/* Factorise a general N x N matrix A into,
 *
 *   P A = L U
 *
 * where P is a permutation matrix, L is unit lower triangular and U
 * is upper triangular.
 *
 * L is stored in the strict lower triangular part of the input
 * matrix. The diagonal elements of L are unity and are not stored.
 *
 * U is stored in the diagonal and upper triangular part of the
 * input matrix.  
 * 
 * P is stored in the permutation p. Column j of P is column k of the
 * identity matrix, where k = permutation->data[j]
 *
 * signum gives the sign of the permutation, (-1)^n, where n is the
 * number of interchanges in the permutation. 
 *
 * See Golub & Van Loan, Matrix Computations, Algorithm 3.4.1 (Gauss
 * Elimination with Partial Pivoting).
 */

int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) 
{
    if (A->size1 != A->size2) {
        mexWarnMsgTxt("LU decomposition requires square matrix");
    } else if (p->size != A->size1) {
        mexWarnMsgTxt("permutation length must match matrix size");
    } else {
        const size_t N = A->size1;
        size_t i, j, k;
        *signum = 1;
        gsl_permutation_init (p);

        for (j = 0; j < N - 1; j++) {
            /* Find maximum in the j-th column */

            REAL ajj, max = fabs (gsl_matrix_get (A, j, j));
            size_t i_pivot = j;

            for (i = j + 1; i < N; i++) {
                REAL aij = fabs (gsl_matrix_get (A, i, j));
                if (aij > max) {
                    max = aij;
                    i_pivot = i;
                }
            }

            if (i_pivot != j) {
                gsl_matrix_swap_rows (A, j, i_pivot);
                gsl_permutation_swap (p, j, i_pivot);
                *signum = -(*signum);
            }
            ajj = gsl_matrix_get (A, j, j);
            if (ajj != 0.0) {
                for (i = j + 1; i < N; i++) {
                    REAL aij = gsl_matrix_get (A, i, j) / ajj;
                    gsl_matrix_set (A, i, j, aij);

                    for (k = j + 1; k < N; k++) {
                        REAL aik = gsl_matrix_get (A, i, k);
                        REAL ajk = gsl_matrix_get (A, j, k);
                        gsl_matrix_set (A, i, k, aik - aij * ajk);
                    }
                }
            }
        }

        return GSL_SUCCESS;
    }
}

int gsl_linalg_LU_invert (const gsl_matrix * LU, const gsl_permutation * p,
        gsl_matrix * inverse)
{
    size_t i, n = LU->size1;

    int status = GSL_SUCCESS;

    gsl_matrix_set_identity (inverse);

    for (i = 0; i < n; i++)
    {
        gsl_vector_view c = gsl_matrix_column (inverse, i);
        int status_i = gsl_linalg_LU_svx (LU, p, &(c.vector));

        if (status_i)
            status = status_i;
    }

    return status;
}

double gsl_linalg_LU_det (gsl_matrix * LU, int signum)
{
    size_t i, n = LU->size1;
    double det = (double) signum;
    for (i = 0; i < n; i++) {
        det *= gsl_matrix_get (LU, i, i);
    }

    return det;
}

int gsl_linalg_LU_svx (const gsl_matrix * LU, const gsl_permutation * p, gsl_vector * x)
{
    if (LU->size1 != LU->size2) {
        mexWarnMsgTxt("LU matrix must be square");
    } else if (LU->size1 != p->size) {
        mexWarnMsgTxt("permutation length must match matrix size");
    } else if (LU->size1 != x->size) {
        mexWarnMsgTxt("matrix size must match solution/rhs size");
    } else {
        /* Apply permutation to RHS */
        gsl_permute_vector (p, x);

        /* Solve for c using forward-substitution, L
         * c = P b */
        gsl_blas_dtrsv (CblasLower, CblasNoTrans, CblasUnit, LU, x);

        /* Perform back-substitution, U x = c
         * */
        gsl_blas_dtrsv (CblasUpper, CblasNoTrans, CblasNonUnit, LU, x);

        return GSL_SUCCESS;
    }
}


gsl_permutation * gsl_permutation_alloc (const size_t n)
{
    gsl_permutation * p;

    if (n == 0)
    {
        mexWarnMsgTxt("permutation length n must be positive integer");
    }

    p = (gsl_permutation *) malloc (sizeof (gsl_permutation));

    if (p == 0)
    {
        mexWarnMsgTxt("failed to allocate space for permutation struct");
    }

    p->data = (size_t *) malloc (n * sizeof (size_t));

    if (p->data == 0)
    {
        free (p);         /* exception in constructor, avoid memory leak */

        mexWarnMsgTxt("failed to allocate space for permutation data");
    }

    p->size = n;

    return p;
}

void gsl_permutation_init (gsl_permutation * p)
{
    const size_t n = p->size ;
    size_t i;

    /* initialize permutation to identity */

    for (i = 0; i < n; i++)
    {
        p->data[i] = i;
    }
}

void gsl_permutation_free (gsl_permutation * p)
{
    free (p->data);
    free (p);
}

int gsl_permutation_swap (gsl_permutation * p, const size_t i, const size_t j)
{
    const size_t size = p->size ;

    if (i >= size)
    {
        mexWarnMsgTxt("first index is out of range");
    }

    if (j >= size)
    {
        mexWarnMsgTxt("second index is out of range");
    }

    if (i != j)
    {
        size_t tmp = p->data[i];
        p->data[i] = p->data[j];
        p->data[j] = tmp;
    }

    return GSL_SUCCESS;
}

gsl_matrix* gsl_matrix_alloc(const size_t n1, const size_t n2)
{
    gsl_block * block;
    gsl_matrix * m;

    if (n1 == 0) {
        mexWarnMsgTxt("matrix dimension n1 must be positive integer");
    } else if (n2 == 0) {
        mexWarnMsgTxt("matrix dimension n2 must be positive integer");
    }
    m = (gsl_matrix *) malloc (sizeof (gsl_matrix));
    if (m == 0) {
        mexWarnMsgTxt("failed to allocate space for matrix struct");
    }

    /* FIXME: n1*n2 could overflow for large dimensions */

    block = gsl_block_alloc(n1 * n2) ;

    if (block == 0) {
        mexWarnMsgTxt("failed to allocate space for block");
    }

    m->data = block->data;
    m->size1 = n1;
    m->size2 = n2;
    m->tda = n2;
    m->block = block;
    m->owner = 1;

    return m;
}

gsl_matrix * gsl_matrix_calloc(const size_t n1, const size_t n2)
{
    size_t i;
    gsl_matrix * m = gsl_matrix_alloc(n1, n2);
    if (m == 0) return 0;

    /* initialize matrix to zero */
    for (i = 0; i < MULTIPLICITY * n1 * n2; i++) {
        m->data[i] = 0;
    }
    return m;
}


void gsl_matrix_free(gsl_matrix * m)
{
    if (m->owner) {
        gsl_block_free(m->block);
    }

    free (m);
}


_gsl_matrix_view gsl_matrix_view_array(ATOMIC * array,
        const size_t n1, const size_t n2)
{
    _gsl_matrix_view view= NULL_MATRIX_VIEW;

    if (n1 == 0) {
        warning("matrix dimension n1 must be positive integer");
    } else if (n2 == 0) {
        warning("matrix dimension n2 must be positive integer");
    }

    {
        gsl_matrix m = NULL_MATRIX;

        m.data = (ATOMIC *)array;
        m.size1 = n1;
        m.size2 = n2;
        m.tda = n2;
        m.block = 0;
        m.owner = 0;

        ((_gsl_matrix_view *)&view)->matrix = m;
        return view;
    }
}


int gsl_matrix_swap_rows(gsl_matrix * m,
        const size_t i, const size_t j)
{
    const size_t size1 = m->size1;
    const size_t size2 = m->size2;

    if (i >= size1) {
        mexWarnMsgTxt("first row index is out of range");
    }
    if (j >= size1) {
        mexWarnMsgTxt("second row index is out of range");
    }

    if (i != j) {
        ATOMIC *row1 = m->data + MULTIPLICITY * i * m->tda;
        ATOMIC *row2 = m->data + MULTIPLICITY * j * m->tda;

        size_t k;

        for (k = 0; k < MULTIPLICITY * size2; k++)
        {
            ATOMIC tmp = row1[k] ;
            row1[k] = row2[k] ;
            row2[k] = tmp ;
        }
    }

    return GSL_SUCCESS;
}

BASE gsl_matrix_get(const gsl_matrix * m, const size_t i, const size_t j)
{
    BASE zero = ZERO;

    if (gsl_check_range) {
        if (i >= m->size1)        /* size_t is unsigned, can't be negative */
            mexWarnMsgTxt("first index out of range");
        else if (j >= m->size2)   /* size_t is unsigned, can't be negative */
            mexWarnMsgTxt("second index out of range");
    }
    return *(BASE *) (m->data + MULTIPLICITY * (i * m->tda + j));
}

void gsl_matrix_set(gsl_matrix * m,
        const size_t i, const size_t j,
        const BASE x)
{
    if (gsl_check_range) {
        if (i >= m->size1)        /* size_t is unsigned, can't be negative */
        {
            mexWarnMsgTxt("first index out of range");
        } else if (j >= m->size2)   /* size_t is unsigned, can't be negative */
        {
            mexWarnMsgTxt("second index out of range");
        }
    }
    *(BASE *) (m->data + MULTIPLICITY * (i * m->tda + j)) = x;
}

void gsl_matrix_set_identity(gsl_matrix * m)
{
    size_t i, j;
    ATOMIC * const data = m->data;
    const size_t p = m->size1 ;
    const size_t q = m->size2 ;
    const size_t tda = m->tda ;

    const BASE zero = ZERO;
    const BASE one = ONE;

    for (i = 0; i < p; i++)
    {
        for (j = 0; j < q; j++)
        {
            *(BASE *) (data + MULTIPLICITY * (i * tda + j)) = ((i == j) ? one : zero);
        }
    }
}
_gsl_vector_view gsl_matrix_column(gsl_matrix * m, const size_t j)
{
    _gsl_vector_view view = NULL_VECTOR_VIEW;

    if (j >= m->size2) {
        mexWarnMsgTxt("column index is out of range");
    }

    gsl_vector v = NULL_VECTOR;

    v.data = m->data + j * MULTIPLICITY;
    v.size = m->size1;
    v.stride = m->tda;
    v.block = m->block;
    v.owner = 0;

    ((_gsl_vector_view *)&view)->vector = v;
    return view;
}

gsl_block* gsl_block_alloc(const size_t n)
{
    gsl_block * b;
    if (n == 0) {
        mexWarnMsgTxt ("block length n must be positive integer");
    }
    b = (gsl_block *) malloc (sizeof (gsl_block));
    if (b == 0) {
        mexWarnMsgTxt("failed to allocate space for block struct");
    }
    b->data = (ATOMIC *) malloc (MULTIPLICITY * n * sizeof (ATOMIC));
    if (b->data == 0) {
        free (b);         /* exception in constructor, avoid memory leak */
        mexWarnMsgTxt("failed to allocate space for block data");
    }
    b->size = n;
    return b;
}

void gsl_block_free(gsl_block * b)
{
    free (b->data);
    free (b);
}

int gsl_permute_vector (const gsl_permutation * p, gsl_vector* v)
{
    if (v->size != p->size) {
        mexWarnMsgTxt("vector and permutation must be the same length");
    }

    gsl_permute(p->data, v->data, v->stride, v->size) ;

    return GSL_SUCCESS;
}

int gsl_permute(const size_t * p, ATOMIC * data, const size_t stride, const size_t n)
{
    size_t i, k, pk;
    for (i = 0; i < n; i++) {
        k = p[i];
        while (k > i) k = p[k];
        if (k < i) continue ;

        /* Now have k == i, i.e the least in
         * its cycle */
        pk = p[k];
        if (pk == i) continue ;

        /* shuffle the elements
         * of the cycle */
        {
            unsigned int a;
            ATOMIC t[MULTIPLICITY];
            for (a = 0; a < MULTIPLICITY; a++)
                t[a] = data[i*stride*MULTIPLICITY + a];

            while (pk != i) {
                for (a = 0; a < MULTIPLICITY; a++) {
                    ATOMIC r1 = data[pk*stride*MULTIPLICITY + a];
                    data[k*stride*MULTIPLICITY + a] = r1;
                }
                k = pk;
                pk = p[k];
            };

            for (a = 0; a < MULTIPLICITY; a++)
                data[k*stride*MULTIPLICITY + a] = t[a];
        }
    }
    return GSL_SUCCESS;
}


int gsl_blas_dtrsv (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
        CBLAS_DIAG_t Diag, const gsl_matrix * A, gsl_vector * X)
{
    const size_t M = A->size1;
    const size_t N = A->size2;
    if (M != N) {
        mexWarnMsgTxt("matrix must be square");
    }
    else if (N != X->size) {
        mexWarnMsgTxt("invalid length");
    }

    cblas_dtrsv (CblasRowMajor, Uplo, TransA, Diag, INT (N), A->data,
            INT (A->tda), X->data, INT (X->stride));
    return GSL_SUCCESS;
}

void
cblas_dtrsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
        const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
        const int N, const double *A, const int lda, double *X,
        const int incX)
{
#define BASE double
#include "source_trsv_r.c"
#undef BASE
}


