MEX = /opt/MATLAB/R2018b/bin/mex
# MEX_FLAGS= -g
# MEX_INCLUDE = -I/home/pablo/gsl/include

smooth: SmoothLDS.c gsl_utils.c
	    ${MEX} ${MEX_FLAGS} ${MEX_INCLUDE} ${LIBRARY} \
	    	SmoothLDS.c utils.c gsl_utils.c mchol.c
