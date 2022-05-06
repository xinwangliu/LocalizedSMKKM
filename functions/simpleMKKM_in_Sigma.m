function [Hstar,Sigma,obj] = simpleMKKM_in_Sigma(KH,numclass,Sigma,option)
% Updated at 2021.12.29 23.00 By Zhangyi
numker = size(KH,3);
% Sigma = ones(numker,1)/numker;

% KHP = zeros(num,num,numker);
% for p = 1:numker
%     KHP(:,:,p) = myLocalKernel(KH,tau,p);
% end
% KH = KHP;
% clear KHP
%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-4;
end
if ~isfield(option,'goldensearchmax')
    option.goldensearchmax=1e-3;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end

% option.seuildiffsigma = min(1/numker * 1e-2 ,1e-3) ;
option.seuildiffsigma = 1/numker * 1e-4  ;
% option.goldensearch_deltmax = 1e-3;

nloop = 1;
loop = 1;
% goldensearch_deltmaxinit = option.goldensearch_deltmax;
%%---
% MaxIter = 30;
% res_mean = zeros(4,MaxIter);
% res_std = zeros(4,MaxIter);
%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Kmatrix = sumKbeta(KH,Sigma.^2);
[Hstar,obj1]= mykernelkmeans(Kmatrix,numclass);
obj(nloop) = obj1;
% [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
[grad] = simpleMKKMGrad(KH,Hstar,Sigma);
Sigmaold  = Sigma;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop
    nloop = nloop+1;
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = simpleMKKMupdate(KH,Sigmaold,grad,obj(nloop-1),Hstar,numclass,option);
    % [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
    %     %-------------------------------
    %     % Numerical cleaning
    %     %-------------------------------
    %    Sigma(find(abs(Sigma<option.numericalprecision)))=0;
    %    Sigma = Sigma/sum(Sigma);
    
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------


    if max(abs(Sigma-Sigmaold))<option.seuildiffsigma || (nloop>2 && (obj(nloop-1)-obj(nloop))/obj(nloop)<1e-5 )
            loop = 0;
%             fprintf(1,'variation convergence criteria reached \n');
    end
    [grad] = simpleMKKMGrad(KH,Hstar,Sigma);
    
        
%     if max(abs(Sigma-Sigmaold))<option.seuildiffsigma &&...
%             option.goldensearch_deltmax > option.goldensearchmax
%         option.goldensearch_deltmax = option.goldensearch_deltmax/10;
%     elseif max(abs(Sigma-Sigmaold))<option.seuildiffsigma &&...
%             option.goldensearch_deltmax <= option.goldensearchmax
%             loop = 0;
%             fprintf(1,'variation convergence criteria reached \n');
%         %     elseif max(abs(Beta-Betaold))>option.seuildiffsigma
%         %     elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
%         %         option.goldensearch_deltmax = option.goldensearch_deltmax*10;
%     end
%     [grad] = simpleMKKMGrad(KH,Hstar,Sigma);


    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
%     if  max(abs(Sigma-Sigmaold))<option.seuildiffsigma
%         loop = 0;
%         fprintf(1,'variation convergence criteria reached \n');
%     end
    
  
    
    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------
    Sigmaold  = Sigma;
end