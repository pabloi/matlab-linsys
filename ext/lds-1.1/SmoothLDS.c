/*  SmoothLDS.c - Kalman Smoother for Linear Dyamical System (LDS)
    Copyright (C) 2005 Philip N. Sabes, Sen Cheng

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

/**************************************************************
*
*   Kalman Smoother for Linear Dyamical System (LDS)
*   P Sabes, based heavily on code by Z Ghahramani
*   Sen Cheng: sped up code (x5)
*
*************************************************************/

/*
* Kalman smoother
*
* returns: [Lik,Xfin,Vfin,V1,Sums]
*     Sums is a structure with elements:
*       XX XY YY UU XU XX1 XU1 xxI xxF
*/

#include <stdlib.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"
#include "gsl_utils.h"
#include "utils.h"

#define zeros(X,Y) mxCreateDoubleMatrix((X),(Y),mxREAL)
#define sendup(X) mexPutVariable("global",#X,X)
void mxOuterSum(const mxArray *A, int M, int t1, int t2,
        const mxArray *B, int N, int s1, int s2,
        mxArray *R)
{
    outerSum(mxGetPr(A), M, t1, t2, mxGetPr(B), N, s1, s2, mxGetPr(R));
}

static mxArray  *A, *B, *C, *D, *Q, *R, *x0, *V0;
static double  *prA, *prB, *prC, *prD, *prQ, *prR, *prx0, *prV0;
static mxArray  *Lik, *Xfin, *Vfin, *V1, *Sums;
static double  *prLik;
static int   *T;
static int      Nx,Ny,Nu,Nw,Ne;

/******************************************************/
void SmoothLDS(
			   const mxArray *Y,
			   const mxArray *U,
			   const mxArray *W
			   )
{
    gsl_matrix *invRp, *Rptmp;
    gsl_permutation *permy= gsl_permutation_alloc(Ny);
    double      *prRptmp;
	mwSize       Vdims[3];
	mxArray   *Xpre, *Xcur, *Vpre, *Vcur;
	double   *prXpre, *prXcur, *prVpre, *prVcur;
	mxArray   *eU,*eY,*eW,*eXfin,*eVfin, *eV1;
	double   *preU,*preY,*preW,*preXfin,*preVfin, *preV1;
	mxArray   *Rp, *K, *innov, *J;
	double   *prRp, *prinvRp, *prK, *prinnov, *prJ;
	int       j,t,e,maxT,sumT;
	double    detiRp,temp;
	const double  lognorm = -0.5*(double)Ny*log(2.0*3.14159265358979);

	gsl_matrix *mx1= gsl_matrix_calloc(Nx,1);
	gsl_matrix *mxx= gsl_matrix_calloc(Nx,Nx);
	gsl_matrix *mxx2= gsl_matrix_calloc(Nx,Nx);
	gsl_matrix *mxx3= gsl_matrix_calloc(Nx,Nx);
	gsl_matrix *mxy= gsl_matrix_calloc(Nx,Ny);
	gsl_matrix *my1= gsl_matrix_calloc(Ny,1);
	double *tx1= mx1->data;
	double *txx2= mxx2->data;
	double *txx3= mxx3->data;
	double *txx= mxx->data;
    double *txy= mxy->data;
	double *ty1= my1->data;

    gsl_matrix_view gxx= gsl_matrix_view_array(txx,Nx,Nx);
    gsl_matrix_view gxx2= gsl_matrix_view_array(txx2,Nx,Nx);
    gsl_permutation *permx= gsl_permutation_alloc(Nx);

	*prLik= 0.0;
	Rp    = zeros(Ny,Ny);
    invRp = gsl_matrix_calloc(Ny,Ny);
	prinvRp= invRp->data;
    Rptmp = gsl_matrix_calloc(Ny,Ny);
	prRptmp= Rptmp->data;
	K     = zeros(Nx,Ny);
	innov = zeros(Ny,1);
	J     = zeros(Nx,Nx);

	prRp= mxGetPr(Rp);
	prK= mxGetPr(K);
	prinnov= mxGetPr(innov);
	prJ= mxGetPr(J);

	Vdims[0]=Nx;  Vdims[1]=Nx;

	/********* Loop On Runs *******
	*
	* Y, U, and all the Xs and Vs are cell arrays of length Ne
	* Ne is the number of "experiments" or simuluation runs.
	*
	* The last dimension on the ith cell element has length T[i]
	*
	*/
	maxT = -1;
	sumT = 0;
	for(e=0;e<Ne;e++){
		sumT += T[e];
		Vdims[2]=T[e];

		/* Data Structure Retrieval and Allocation */
		eY = mxGetCell(Y,e); preY= mxGetPr(eY);
		if(Nu>0) { eU = mxGetCell(U,e); preU= mxGetPr(eU); }
		if(Nw>0) { eW = mxGetCell(W,e); preW= mxGetPr(eW); }

		eXfin = zeros(Nx,T[e]);
		eVfin = mxCreateNumericArray(3,Vdims,mxDOUBLE_CLASS,mxREAL);
		eV1   = mxCreateNumericArray(3,Vdims,mxDOUBLE_CLASS,mxREAL);
        preXfin= mxGetPr(eXfin);
        preVfin= mxGetPr(eVfin);
        preV1= mxGetPr(eV1);
/*        printf("A");*/
		/* Need to Enlarge the Data Structures */
		if(T[e]>maxT){
			if(e>0){
				/* Destroy Old Arrays */
				mxDestroyArray(Xpre);
				mxDestroyArray(Xcur);
				mxDestroyArray(Vpre);
				mxDestroyArray(Vcur);
			}
			/* Make New Ones */
			Xpre = zeros(Nx,T[e]);
			Xcur = zeros(Nx,T[e]);
			Vpre = mxCreateNumericArray(3,Vdims,mxDOUBLE_CLASS,mxREAL);
			Vcur = mxCreateNumericArray(3,Vdims,mxDOUBLE_CLASS,mxREAL);
            prXpre= mxGetPr(Xpre);
            prXcur= mxGetPr(Xcur);
            prVpre= mxGetPr(Vpre);
            prVcur= mxGetPr(Vcur);
			maxT=T[e];
/*        printf("B");*/
		}

		/************************ FORWARD PASS *************************/

		*prLik += T[e]*lognorm;

		for (t=1;t<=T[e];t++){

/*        printf("C");*/
			/****** State Update *****/
			if(t>1){
				/* Xp(:,t) = A*Xc(:,t-1) + B*U{e}(:,t-1); */
				submult3d(prA,1, prXcur,t-1, txx, Nx,Nx,1);
				subcopy(prXpre,t, txx,1, Nx);
				if(Nu>0){
					submult3d(prB,1, preU,t-1, tx1, Nx,Nu,1);
					subinc(prXpre,t, tx1,1, Nx );
				}

				/* Vp(:,:,t)= Q + A*Vc(:,:,t-1)*A'; */
				submult3dtr(prVcur,t-1,prA,1,txx2,Nx,Nx,Nx); /*xx2=Vc(:,:,t-1)*A'*/
				multm(prA,txx2,txx,Nx,Nx,Nx);      /*xx= A*Vc(:,:,t-1)*A' */
				subinc(txx,1, prQ,1, Nx*Nx);      /* xx= Q + A*Vc(:,:,t-1)*A' */
				subcopy(prVpre,t, txx,1, Nx*Nx);
			} else{
				subcopy(prXpre,1,prx0,1,Nx);
				subcopy(prVpre,1,prV0,1,Nx*Nx);
			}
      /* Determining if next sample is missing */
      subcopy(prinnov,1, preY,t, Ny);
      bool missingFlag = false;
      for (int i=1;i<=Ny;i++){
        missingFlag= missingFlag || isnan(prinnov[i]);
      }

      if(!missingFlag){
			/****** Kalman Gain *****/
			/* invRp = inv( C*Vp(:,:,t)*C' + R ); */
			submult3dtr(prVpre,t, prC,1, txy, Nx,Nx,Ny);   /* xy= Vp*C' */
			multm(prC,txy,prRp,Ny,Nx,Ny);  /* Rp= C*Vp*C' */
			subinc(prRp,1, prR,1, Ny*Ny);      /* Rp= C*Vp*C'+R */

            for(j=0; j<Ny*Ny; j++) prRptmp[j]= prRp[j]; /* @@ tmp */
			myinv(Ny,Rptmp,invRp,permy,&detiRp); 	      /* Rp= inv(C*Vp*C'+R) */

			/* K = Vp(:,:,t) * C' * invRp; */
			multm(txy,prinvRp,prK,Nx,Ny,Ny);

			/****** Innovation *****/
			/* innov = Y{e}(:,t)-C*Xp(:,t); */
			submult3d(prC,1, prXpre,t, ty1, Ny,Nx,1);
			subdec(prinnov,1, ty1,1, Ny);
			if(Nw>0){
				/* innov = Y{e}(:,t)-C*Xp(:,t)-LDS.D*W{e}(:,t); */
				submult3d(prD,1, preW,t, ty1, Ny,Nw,1);
				subdec(prinnov,1, ty1,1, Ny);
			}

			/* Xc(:,t)   = Xp(:,t) + K*innov; */
			subcopy(prXcur,t, prXpre,t, Nx);
			multm(prK,prinnov,tx1,Nx,Ny,1);
			subinc(prXcur,t, tx1,1, Nx);

			/* Vc(:,:,t) = (I-K*C)*Vp(:,:,t); */
			multm(prK,prC,txx2,Nx,Ny,Nx); /* xx2= K*C */
			eyem(txx3, Nx);
			subdec(txx3,1, txx2,1, Nx*Nx);  /* xx3= (I-K*C) */
			submult3d(txx3,1, prVpre,t, txx, Nx,Nx,Nx); /*xx=(I-KC)Vp*/
			subcopy(prVcur,t, txx,1, Nx*Nx);

			/****** Likelihood *****/
			/* Lik +=  0.5*log(detInvRp) - 0.5*innov'*invRp*innov; */
			multm(prinvRp,prinnov,ty1,Ny,Ny,1);
			*prLik += -0.5*log(detiRp) - 0.5*innerProd(prinnov,ty1,Ny);
    }else{
      /* printf("Missing sample detected at (t=%d)",t);
      printf("State %d",prXpre[t]); */
      subcopy(prXcur,t, prXpre,t, Nx);
      subcopy(prVcur,t, prVpre,t, Nx*Nx);
    }
		}

/*        printf("C2");*/

		/************************ BACKWARD PASS *************************/
		/* mexPrintf("backward pass!\n"); */

		/* Xe(:,T(e))    = Xc(:,T(e));   */
		/* Ve(:,:,T(e))  = Vc(:,:,T(e)); */
		t=T[e];
/*        printf("(t=%d)",t);*/
        if(t > 0) {
            subcopy(preXfin,t, prXcur,t, Nx);
            subcopy(preVfin,t, prVcur,t, Nx*Nx);
        }

/*        printf("D");*/
		for (t=T[e]-1;t>0;t--)
		{
			/* J = Vc(:,:,t)*A'*inv(Vp(:,:,t+1));  */
			subcopy(txx,1, prVpre,t+1, Nx*Nx); /* xx = Vp */
/*            inv(mxx,mxx2,&temp);  |+ xx2 = inv(Vp) +|*/
			myinv(Nx,&gxx.matrix,&gxx2.matrix,permx,&temp); 	  /* xx2 = inv(Vp) */
			submult3dtr(prVcur,t,prA,1,txx,Nx,Nx,Nx); /* xx= Vc*A' */
			multm(txx,txx2,prJ,Nx,Nx,Nx);

			/* Xe(:,t) = Xc(:,t) + J*(Xe(:,t+1)-Xp(:,t+1)); */
			subcopy(preXfin,t, prXcur,t, Nx); /* Xe= Xc */
			submult3d(prJ,1, preXfin,t+1, tx1, Nx,Nx,1);/*x1=J*Xe*/
			subinc(preXfin,t, tx1,1, Nx); /* +J*Xe */
			submult3d(prJ,1, prXpre,t+1, tx1, Nx,Nx,1);/*x1=J*Xp*/
			subdec(preXfin,t, tx1,1, Nx); /* -JXp */

			/* Ve(:,:,t) = Vc(:,:,t) + J*(Ve(:,:,t+1)-Vp(:,:,t+1))*J'; */
			subcopy(preVfin,t, prVcur,t, Nx*Nx);
			zerosm(txx3, Nx,Nx);
			subinc(txx3,1, preVfin,t+1, Nx*Nx);
			subdec(txx3,1, prVpre,t+1, Nx*Nx); /* xx3= Ve-Vp */
			multm(prJ,txx3,txx2,Nx,Nx,Nx);     /* xx2= J*(Ve-Vp) */
			multmtr(txx2, prJ, txx,Nx,Nx,Nx);/* xx= J*(Ve-Vp)*J' */
			subinc(preVfin,t, txx,1, Nx*Nx);

			/* V1e(:,:,t+1) = Ve(:,:,t+1) * J'; */
			submult3dtr(preVfin,t+1, prJ,1, txx, Nx,Nx,Nx);
			subcopy(preV1,t+1, txx,1, Nx*Nx);

		}

		mxSetCell(Xfin, e, eXfin);
		mxSetCell(Vfin, e, eVfin);
		mxSetCell(V1, e, eV1);
/*        printf("E");*/
  } /* for e */

*prLik /= sumT;
   /*printf("%f\t%d\n", pLik, sumT);*/

  /* Memory Management */
    if (e>0) {
        mxDestroyArray(Xpre);
        mxDestroyArray(Xcur);
        mxDestroyArray(Vpre);
        mxDestroyArray(Vcur);
    }
/*        printf("F");*/

  mxDestroyArray(Rp);
  gsl_matrix_free(invRp);
  gsl_matrix_free(Rptmp);
  mxDestroyArray(K);
  mxDestroyArray(innov);
  mxDestroyArray(J);

  gsl_matrix_free(mx1);
  gsl_matrix_free(mxx);
  gsl_matrix_free(mxx2);
  gsl_matrix_free(mxx3);
  gsl_matrix_free(mxy);
  gsl_matrix_free(my1);
}


/******************************************************/
void makeSums(
			  const mxArray *Y,
			  const mxArray *U,
			  const mxArray *W
			  )
{
	const char *SumsNames[] =
		{ "XX", "YY", "UU", "WW", "XY", "XU", "XW", "YW",
			"XX1", "XU1", "xxI", "xxF"};

	mxArray *XX,*YY,*UU,*WW,*XY,*XU,*XW,*YW,*XX1,*XU1,*xxI,*xxF;
	mxArray *eU,*eY,*eW,*eXfin,*eVfin, *eV1;
	int     e,t,eT;


	/* XX YY UU WW XY XU XW YW XX1 XU1 xxI xxF */
	XX = zeros(Nx,Nx);
	YY = zeros(Ny,Ny);
	XY = zeros(Nx,Ny);
	XX1 = zeros(Nx,Nx);
	xxI = zeros(Nx,Nx);
	xxF = zeros(Nx,Nx);
	if(Nu>0){
		UU = zeros(Nu,Nu);
		XU = zeros(Nx,Nu);
		XU1 = zeros(Nx,Nu);
	}
	if(Nw>0){
		WW = zeros(Nw,Nw);
		XW = zeros(Nx,Nw);
		YW = zeros(Ny,Nw);
	}

	for(e=0;e<Ne;e++){

		eT = T[e];

		/* Data Structure Retrieval and Allocation */
		eY = mxGetCell(Y,e);
		eU = Nu>0 ? mxGetCell(U,e) : 0;
		eW = Nw>0 ? mxGetCell(W,e) : 0;

		eXfin = mxGetCell(Xfin,e);
		eVfin = mxGetCell(Vfin,e);
		eV1   = mxGetCell(V1,e);

		/* YY XY */
		mxOuterSum(eY,Ny,1,eT, eY,Ny,1,eT,YY);
		mxOuterSum(eXfin,Nx,1,eT,  eY,Ny,1,eT,XY);

		/* XX XX1 xxI xxF */
		mxOuterSum(eXfin,Nx,1,eT, eXfin,Nx,1,eT,XX);
		mxOuterSum(eXfin,Nx,2,eT,  eXfin,Nx,1,eT-1,XX1);

		subinc(mxGetPr(XX),1, mxGetPr(eVfin),1, Nx*Nx);
		for(t=2;t<=eT;t++){
			subinc(mxGetPr(XX),1, mxGetPr(eVfin),t, Nx*Nx);
			subinc(mxGetPr(XX1),1, mxGetPr(eV1),t, Nx*Nx);
		}

		mxOuterSum(eXfin,Nx,1,1, eXfin,Nx,1,1,xxI);
		subinc(mxGetPr(xxI),1, mxGetPr(eVfin),1, Nx*Nx);

		mxOuterSum(eXfin,Nx,eT,eT,  eXfin,Nx,eT,eT,xxF);
		subinc(mxGetPr(xxF),1, mxGetPr(eVfin),eT, Nx*Nx);


		/* UU XU XU1 */
		if(Nu>0){
			mxOuterSum(eU,Nu,1,eT-1,  eU,Nu,1,eT-1,UU);

			mxOuterSum(eXfin,Nx,1,eT-1,  eU,Nu,1,eT-1,XU);

			mxOuterSum(eXfin,Nx,2,eT,  eU,Nu,1,eT-1,XU1);
		}


		/* WW XW YW */
		if(Nw>0){
			mxOuterSum(eW,Nw,1,eT,  eW,Nw,1,eT,WW);
			mxOuterSum(eXfin,Nx,1,eT,  eW,Nw,1,eT,XW);
			mxOuterSum(eY,Ny,1,eT,  eW,Nw,1,eT,YW);
		}
	}

/*
	sendup(XX);
	sendup(YY);
	sendup(XY);
	sendup(XX1);
	sendup(xxI);
	sendup(xxF);

	if(Nu>0){
		sendup(UU);
		sendup(XU);
		sendup(XU1);
	}
	if(Nw>0){
		sendup(WW);
		sendup(XW);
		sendup(YW);
	}
*/

	/* XX YY UU WW XY XU XW YW XX1 XU1 xxI xxF */
	Sums = mxCreateStructMatrix(1,1,12,SumsNames);

    mxSetField(Sums,0,"XX",XX);
    mxSetField(Sums,0,"YY",YY);
    mxSetField(Sums,0,"XY",XY);
    mxSetField(Sums,0,"XX1",XX1);
    mxSetField(Sums,0,"xxI",xxI);
    mxSetField(Sums,0,"xxF",xxF);

	if(Nu>0){
		mxSetField(Sums,0,"UU",UU);
		mxSetField(Sums,0,"XU",XU);
		mxSetField(Sums,0,"XU1",XU1);
    }

	if(Nw>0){
		mxSetField(Sums,0,"WW",WW);
		mxSetField(Sums,0,"XW",XW);
		mxSetField(Sums,0,"YW",YW);
	}
}

/******************************************************/
/******************************************************/
/******************************************************/
void mexFunction(
				 int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[]
				 )
{
	const mxArray *LDS, *Y, *U, *W;
	mxArray  *mtmp;
	int    i;
	mwSize    Vdims[3];
	char	 buffer[100];


	/************* Check for proper number of arguments ********/
	/* Inputs:   [LDS, Y, U, W]  or [LDS, Y, U] or [LDS,Y] */
	/* Outputs:  [Lik, X, V, V1, Sums]  */
	if( nrhs!=2 && nrhs!=3 && nrhs!=4 ){
		sprintf(buffer,
			"SmoothLDS: args (LDS,Y,U,V) or (LDS,Y,U) or (LDS,Y). Got %d input args",
			nrhs);
		mexErrMsgTxt(buffer);
	}
	if (nlhs>5)
    {
		sprintf(buffer, "SmoothLDS: outputs [Lik, X, V, V1, Sums]. Got %d outputs",
			nlhs);
		mexErrMsgTxt(buffer);
    }

	i=0;
	LDS = prhs[i++];
	Y   = prhs[i++];
	if(nrhs>=3) U = prhs[i++];
	if(nrhs>=4) W = prhs[i++];

	/************* Unpack LDS and CHECK DIMENSIONS ********/
	if( (A = mxGetField(LDS, 0, "A")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain A");
	if( (C = mxGetField(LDS, 0, "C")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain C");
	if( (Q = mxGetField(LDS, 0, "Q")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain Q");
	if( (R = mxGetField(LDS, 0, "R")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain R");
	if( (x0 = mxGetField(LDS, 0, "x0")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain x0");
	if( (V0 = mxGetField(LDS, 0, "V0")) == NULL )
		mexErrMsgTxt("SmoothLDS: LDS doesn't contain V0");

	prA= mxGetPr(A);
	prC= mxGetPr(C);
	prQ= mxGetPr(Q);
	prR= mxGetPr(R);
	prx0= mxGetPr(x0);
	prV0= mxGetPr(V0);

	/* Ugly Dimension Checking for LDS */
	/* A */
	Nx = mxGetN(A); /* rows */
	if( Nx!=mxGetM(A) ) mexErrMsgTxt("SmoothLDS: A not square");
	/* C */
	Ny = mxGetM(C); /* cols */
	if( Nx!=mxGetN(C) ) mexErrMsgTxt("SmoothLDS: C does not have Nx cols");
	/* B */
	if(nrhs>=3){
		if( (B = mxGetField(LDS, 0, "B")) == NULL )
			mexErrMsgTxt("SmoothLDS: called with input U, but LDS doesn't contain B");
		prB= mxGetPr(B);
		Nu = mxGetN(B); /* cols */
		if( Nu!=0 && Nx!=mxGetM(B) ) mexErrMsgTxt("SmoothLDS: B does not have Nx rows");
	}
	else Nu=0;
	/* D */
	if(nrhs>=4){
		if( (D = mxGetField(LDS, 0, "D")) == NULL )
			mexErrMsgTxt("SmoothLDS: called with input W, but LDS doesn't contain D");
		prD= mxGetPr(D);
		Nw = mxGetN(D); /* cols */
		if( Nw!=0 && Ny!=mxGetM(D) ) mexErrMsgTxt("SmoothLDS: D does not have Ny rows");
	}
	else Nw=0;
	/* Q */
	if( Nx!=mxGetM(Q) || Nx!=mxGetN(Q) )
		mexErrMsgTxt("SmoothLDS: Q is not Nx by Nx");
	/* R */
	if( Ny!=mxGetM(R) || Ny!=mxGetN(R) )
		mexErrMsgTxt("SmoothLDS: R is not Ny by Ny");
	/* x0 */
	if( Nx!=mxGetM(x0) || 1!=mxGetN(x0) )
		mexErrMsgTxt("SmoothLDS: x0 is not Nx by 1");
	/* V0 */
	if( Nx!=mxGetM(V0) || Nx!=mxGetN(V0) )
		mexErrMsgTxt("SmoothLDS: V0 is not Nx by Nx");


	/************* Unpack Y, U and W and CHECK DIMENSIONS *************/
	/* Do all of this here so that you don't have do in the main loop */
	if( !mxIsCell(Y) )
		mexErrMsgTxt("SmoothLDS: mex Version requires Y,U to be cell arrays");

	/* Check Y's Dimensions and Load T */
	Ne  = mxGetM(Y)*mxGetN(Y);
	T  = (int*) mxCalloc(Ne, sizeof(int));
	for(i=0;i<Ne;i++){
		mtmp=mxGetCell(Y,i);
		if(mtmp==NULL)
			mexErrMsgTxt("SmoothLDS: problem reading cell array Y");
		if( Ny!=mxGetM(mtmp) && mxGetM(mtmp)!= 0)
			mexErrMsgTxt("SmoothLDS: an entry in Y does not have Ny rows");
		T[i]=mxGetN(mtmp);
	}

	/* Check U's Dimensions */
	if(Nu>0){
		if( !mxIsCell(U) || Ne!=mxGetM(U)*mxGetN(U) )
			mexErrMsgTxt("SmoothLDS: U is not a cell array with same size as Y");
		for(i=0;i<Ne;i++){
			mtmp=mxGetCell(U,i);
			if(mtmp==NULL)
				mexErrMsgTxt("SmoothLDS: problem reading cell array U");
			if( Nu!=mxGetM(mtmp) && mxGetM(mtmp)!= 0)
				mexErrMsgTxt("SmoothLDS: an entry in U does not have Nu rows");
			if( T[i]!=mxGetN(mtmp) )
				mexErrMsgTxt("SmoothLDS: an entry in U has the wrong length");
		}
	}

	/* Check W's Dimensions */
	if(Nw>0){
		if( !mxIsCell(W) || Ne!=mxGetM(W)*mxGetN(W) )
			mexErrMsgTxt("SmoothLDS: W is not a cell array with same size as Y");
		for(i=0;i<Ne;i++){
			mtmp=mxGetCell(W,i);
			if(mtmp==NULL)
				mexErrMsgTxt("SmoothLDS: problem reading cell array W");
			if( Nw!=mxGetM(mtmp) && mxGetM(mtmp)!= 0)
				mexErrMsgTxt("SmoothLDS: an entry in W does not have Nw rows");
			if( T[i]!=mxGetN(mtmp) )
				mexErrMsgTxt("SmoothLDS: an entry in W has the wrong length");
		}
	}

	/************* Create Output Matrices ********/

	/* Lik */
	Lik = zeros(1,1);
	prLik= mxGetPr(Lik);

	/* Xfin Vfin */
	Vdims[0]=Nx;  Vdims[1]=Nx;
	Xfin = mxCreateCellMatrix(1,Ne);
	Vfin = mxCreateCellMatrix(1,Ne);
	V1   = mxCreateCellMatrix(1,Ne);

	/************* Do Computation and Return ********/

	SmoothLDS(Y,U,W);

	i=0;
	plhs[i++]=Lik;

	if(nlhs>i) plhs[i++]=Xfin;
	else mxDestroyArray(Xfin);

	if(nlhs>i) plhs[i++]=Vfin;
	else mxDestroyArray(Vfin);

	if(nlhs>i) plhs[i++]=V1;
	else mxDestroyArray(V1);

	if(nlhs>i){ makeSums(Y,U,W); plhs[i++]=Sums; }

}
