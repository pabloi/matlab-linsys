MEX = /opt/MATLAB/R2018b/bin/mex  
# MEX_FLAGS= -g
# MEX_INCLUDE = -I/usr/include

smooth: gsl_utils.c SmoothLDS.c
	    ${MEX} ${MEX_FLAGS} ${MEX_INCLUDE} ${LIBRARY} \
	    	SmoothLDS.c gsl_utils.c utils.c
