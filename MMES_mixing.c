#include "mex.h"
#include "matrix.h"
// #include <math.h>

// y = mixing(Q,rndk,rn)
// input Q[D*m], rndk[l1*ms], rn[ms*l1]
// Y = zeros(dim,l1);
// for i = 1 : ms
//     Y = Y + Q(:,rndk(:,i)) .* rn(i,:);
// end
// mex -v COPTIMFLAGS="-Ofast -DNDEBUG" CFLAGS="-march=native" MMES_mixing.c

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double *Q = mxGetPr(prhs[0]);
	size_t dim = mxGetM(prhs[0]);
	size_t m = mxGetN(prhs[0]);

	double *rndk = mxGetPr(prhs[1]);
	size_t l1 = mxGetM(prhs[1]);
	size_t ms = mxGetN(prhs[1]);

	double *rn = mxGetPr(prhs[2]);

	plhs[0] = mxCreateDoubleMatrix(dim, l1, mxREAL);
	double *Y = mxGetPr(plhs[0]);
	// for (int i = 0; i < dim*l1; i++) Y[i] = 0;

	for (int i = 0; i < l1; i++){
		for (int j = 0; j < ms; j++){
			double r = *rn;
			int c = (int) *rndk;
			for (int n = 0; n < dim; n++){
				Y[n + i*dim] += Q[n + (c-1)*dim] * r;
			}
			rn++;
			rndk++;
		}
	}
}
