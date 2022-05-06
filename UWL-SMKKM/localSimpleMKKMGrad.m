function [grad] = localSimpleMKKMGrad(KH,NN,Hstar,Sigma)

d=size(KH,3);
grad=zeros(d,1);
for k=1:d
     grad(k) = 2*Sigma(k)*trace(Hstar'*(KH(:,:,k).*NN)*Hstar);  
end