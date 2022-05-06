function [Beta,Hstar,Sigma,CostNew] = SampleAdaptiveLocalSimpleMKKMupdate(KH,NS,Sigma,Beta,GradNew,CostNew,HstarNew,numclass,option)
% Updated at 2021.12.29 
% Author Xinwang Liu
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
num = size(KH,1);
gold = (sqrt(5)+1)/2 ;
BetaInit = Beta;
BetaNew  = BetaInit;

NormGrad = GradNew'*GradNew;
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
Hstar = HstarNew;
Sigmaold = Sigma;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------
switch option.firstbasevariable
    case 'first'
        [val,coord] = max(BetaNew) ;
    case 'random'
        [val,coord] = max(BetaNew) ;
        coord=find(BetaNew==val);
        indperm=randperm(length(coord));
        coord=coord(indperm(1));
    case 'fullrandom'
        indzero=find(BetaNew~=0);
        if ~isempty(indzero)
            [mini,coord]=min(GradNew(indzero));
            coord=indzero(coord);
        else
            [val,coord] = max(BetaNew) ;
        end
end
GradNew = GradNew - GradNew(coord);
desc = - GradNew.* ( (BetaNew>0) | (GradNew<0) );
desc(coord) = - sum(desc); 

%----------------------------------------------------
% Compute optimal stepsize
%-----------------------------------------------------
stepmin  = 0;
costmin  = CostOld;
costmax  = 0;

%-----------------------------------------------------
% maximum stepsize
%-----------------------------------------------------
ind = find(desc<0);
stepmax = min(-(BetaNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) || stepmax==0
    Beta = BetaNew;
    return
end


[costmax,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmax,desc,BetaInit,numclass,option);
       

%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------
Step = [stepmin stepmax];
Cost = [costmin costmax];
% optimization of stepsize by golden search

coord = 0;
while (stepmax-stepmin)>option.goldensearch_deltmax*(abs(deltmax)) && stepmax > eps
    
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedr,desc,BetaInit,numclass,option);
            [costmedl,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedl,desc,BetaInit,numclass,option);
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            costmedr = costmedl;
            [costmedl,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedl,desc,BetaInit,numclass,option);
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            costmedl = costmedr;
            [costmedr,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedr,desc,BetaInit,numclass,option);
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedr,desc,BetaInit,numclass,option);
            [costmedl,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedl,desc,BetaInit,numclass,option);
        otherwise
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedr,desc,BetaInit,numclass,option);
            [costmedl,~,~] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,stepmedl,desc,BetaInit,numclass,option);
            %init
    end
    
    
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [~,coord] = min(Cost);
    
end

%---------------------------------
% Final Updates
%---------------------------------
[~,coord] = min(Cost);
CostNew = Cost(coord);
step = Step(coord);
% Sigma update
if CostNew < CostOld
    [CostNew,Hstar,Sigma] = SampleAdaptiveLocalCostSimpleMKKM(KH,NS,Sigmaold,step,desc,BetaInit,numclass,option);
    Beta = BetaNew + step * desc;
    Beta(Beta<option.numericalprecision)=0;
    Beta = num*Beta/sum(Beta);
else
    Hstar = HstarNew;
    Sigma = Sigmaold;
    Beta = BetaInit;
    CostNew = CostOld;
end