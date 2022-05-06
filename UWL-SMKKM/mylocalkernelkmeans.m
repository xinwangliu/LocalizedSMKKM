function [H,obj]= mylocalkernelkmeans(K,A0,cluster_count)

opt.disp = 0;
K0 = A0.*K;
K0= (K0+K0')/2;
[H,~] = eigs(K0,cluster_count,'LA',opt);
obj = trace(H'*K0*H);