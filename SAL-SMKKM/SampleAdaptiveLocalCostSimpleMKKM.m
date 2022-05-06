function [cost,Hstar,Sigma] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigma,StepBeta,DirBeta,Beta,numclass,option)
% Updated at 2021.12.29 23.00 By Zhangyi
global nbcall
nbcall=nbcall+1;
num  = size(KH,1);
numK = size(KH,3);

Beta = Beta + StepBeta * DirBeta;

Beta(Beta<option.numericalprecision)=0;
Beta = num*Beta/sum(Beta);



Amatrix = zeros(num);
sqrtBeta = sqrt(Beta);
for i =1:num
    Amatrix(NS(:,i),i) = sqrtBeta(i);
end
Amatrix = Amatrix * Amatrix';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KHatmatrix = zeros(num,num,numK);
for q = 1:numK
    KHatmatrix(:,:,q) = KH(:,:,q).*Amatrix;
    KHatmatrix(:,:,q) = (KHatmatrix(:,:,q)+KHatmatrix(:,:,q)')/2;
end

[Hstar,Sigma,obj1] = simpleMKKM_in_Sigma(KHatmatrix,numclass,Sigma,option);
cost = obj1(end);