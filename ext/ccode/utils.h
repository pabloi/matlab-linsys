/*  utils.h - 
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

#ifndef UTILS_H
#define UTILS_H


void subcopy(double *A, int k, const double *B, int j, int n);
void subinc(double *A, int k, const double *B, int j, int n);
void subdec(double *A, int k, const double *B, int j, int n);
int  disp(const double *A);

void multm(const double *A, const double *B, double *C, int L, int M, int N);
void multmtr(const double *A, const double *B, double *C, int L, int M, int N);
void submult3dtr(const double *A, int k, const double *B, int j, 
		double *R, int L, int M, int N);
void submult3d(const double *A, int k, const double *B, int j, 
		double *R, int L, int M, int N);
void eyem(double *R, int r);
void zerosm(double *R, int n, int m);
double   innerProd (const double *A, const double *B, int N);


void outerSum(const double *A, int M, int t1, int t2, const double *B, int N, int s1, int s2, double *R);

 
#endif

