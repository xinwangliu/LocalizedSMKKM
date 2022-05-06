function [Hstar,Sigma,Beta,obj] = SampleAdaptiveLocalSimpleMKKM(KH,tau,numclass,option)
% Updated at 2021.12.29
% TPAMI 2022  Submission

if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-2;
end
if ~isfield(option,'goldensearchmax')
    option.goldensearchmax=1e-6;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end

goldensearch_deltmaxinit = option.goldensearch_deltmax;

num = size(KH,1);
numK = size(KH,3);
Sigma = ones(numK,1)/numK;
Kmatrix = sumKbeta(KH,Sigma.^2);
Beta =  ones(num,1);
numSel = round(tau*num);
NS = genarateNeighborhood(Kmatrix,numSel);
%%--Calculate Neighborhood--%%%%%%
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
obj(1) = obj1(end);

[grad] = SampleAdaptiveLocalSimpleMKKMGrad(KH,NS,Hstar,Sigma,Beta);
Betaold  = Beta;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%
clag = 0;
nloop = 1;
loop = 1;
while loop
    nloop = nloop+1;
	[Beta,Hstar,Sigma,obj(nloop)] = SampleAdaptiveLocalSimpleMKKMupdate(KH,NS,Sigma,Betaold,grad,obj(nloop-1),Hstar,numclass,option);

    
    if max(abs(Beta-Betaold))<option.seuildiffsigma || (nloop>2 && (obj(nloop-1)-obj(nloop))/obj(nloop)<1e-3 )
        loop = 0;
        fprintf(1,'variation convergence criteria reached \n');
    end
    [grad] = SampleAdaptiveLocalSimpleMKKMGrad(KH,NS,Hstar,Sigma,Beta);
    Betaold  = Beta;
end