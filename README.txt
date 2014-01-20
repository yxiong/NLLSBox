================================================================
NLLSBox --- A matlab toolbox for nonlinear least squares problem.
================================================================

Author: Ying Xiong.
Created: Jan 20, 2014.

================================================================
Quick start.
================================================================
>> addpath('Utils');
>> NLLSBoxTest;
>> demoNLLSBox;

The main function of the package is 'NonlinearLeastSquares.m', which has a
similar interface as Matlab's 'lsqnonlin' function (unless otherwise
stated). See "help NonlinearLeastSquares" for more details.

================================================================
Notation and convention.
================================================================

The cost function we will minimize is
  F(x) = \sum_{i=1}^M f_i(x)^2
where 'x' is a vector of dimension N, 'f' is a vector function of dimension M,
and 'F' is a scalar. We also define 'J' as the Jacobian matrix of function 'f',
which is a matrix of dimension MxN.

All vectors are column vectors unless otherwise stated.

See 'NonlinearLeastSquares.pdf' for a more detailed documentation.
