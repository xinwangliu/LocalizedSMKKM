function [grad] = SampleAdaptiveLocalSimpleMKKMGrad(KH,NS,Hstar,Sigma,Beta)

num = size(KH,1);
Kmatrix = sumKbeta(KH,Sigma.^2);
Kmatrix = (Kmatrix + Kmatrix')/2;

grad=zeros(num,1);

Cmatrix = (Hstar*Hstar') .* Kmatrix;
parfor i =1:num
    grad(i) = sum(sum(Cmatrix(NS(:,i),NS(:,i))));
end

end