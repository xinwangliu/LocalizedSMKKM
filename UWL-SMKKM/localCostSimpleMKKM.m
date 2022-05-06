function [cost,Hstar] = localCostSimpleMKKM(KH,StepSigma,DirSigma,NN,Sigma,numclass)

global nbcall
nbcall=nbcall+1;
Sigma = Sigma+ StepSigma * DirSigma;
Kmatrix = sumKbeta(KH,(Sigma.*Sigma));
[Hstar,cost]= mylocalkernelkmeans(Kmatrix,NN,numclass);