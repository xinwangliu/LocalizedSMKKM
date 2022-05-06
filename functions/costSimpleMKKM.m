function [cost,Hstar] = costSimpleMKKM(KH,StepSigma,DirSigma,Sigma,numclass,option)
% Updated at 2021.12.29 23.00 By Zhangyi
global nbcall
nbcall=nbcall+1;

Sigma = Sigma+ StepSigma * DirSigma;
Sigma(Sigma<option.numericalprecision)=0;
Sigma=Sigma/sum(Sigma);

Kmatrix = sumKbeta(KH,(Sigma.*Sigma));
[Hstar,cost]= mykernelkmeans(Kmatrix,numclass);