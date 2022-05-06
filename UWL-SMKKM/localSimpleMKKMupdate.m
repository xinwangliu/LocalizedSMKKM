function [Sigma,Hstar,CostNew] = localSimpleMKKMupdate(KH,Sigma,GradNew,NN,CostNew,numclass,option)
% Updated 2021.03
% Author Xinwang Liu
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
gold = (sqrt(5)+1)/2 ;
SigmaInit = Sigma ;
SigmaNew  = SigmaInit; 

NormGrad = GradNew'*GradNew;
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------
switch option.firstbasevariable
    case 'first'
        [val,coord] = max(SigmaNew) ;
    case 'random'
        [val,coord] = max(SigmaNew) ;
        coord=find(SigmaNew==val);
        indperm=randperm(length(coord));
        coord=coord(indperm(1));
    case 'fullrandom'
        indzero=find(SigmaNew~=0);
        if ~isempty(indzero)
            [mini,coord]=min(GradNew(indzero));
            coord=indzero(coord);
        else
            [val,coord] = max(SigmaNew) ;
        end
end
GradNew = GradNew - GradNew(coord);
desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) );
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
stepmax = min(-(SigmaNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) || stepmax==0
    Sigma = SigmaNew;
    return
end
if stepmax > 0.1
     stepmax=0.1;
end

%-----------------------------------------------------
%  Projected gradient
%-----------------------------------------------------

while costmax<costmin
    [costmax,Hstar] = localCostSimpleMKKM(KH,stepmax,desc,NN,SigmaNew,numclass);
    
    if costmax<costmin
        costmin = costmax;
        SigmaNew  = SigmaNew + stepmax * desc;
		%-------------------------------
		% Numerical cleaning
		%-------------------------------
		SigmaNew(find(abs(SigmaNew<option.numericalprecision)))=0;
		SigmaNew=SigmaNew/sum(SigmaNew);
        % project descent direction in the new admissible cone
        % keep the same direction of descent while cost decrease
        %desc = desc .* ( (SigmaNew>0) | (desc>0) ) ;
        desc = desc .* ( (SigmaNew>option.numericalprecision)|(desc>0));
        desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));  
        ind = find(desc<0);
        if ~isempty(ind)
            stepmax = min(-(SigmaNew(ind))./desc(ind));
            deltmax = stepmax;
            costmax = 0;
        else
            stepmax = 0;
            deltmax = 0;
        end      
    end
end

%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------
Step = [stepmin stepmax];
Cost = [costmin costmax];
[val,coord] = min(Cost);
% optimization of stepsize by golden search
while (stepmax-stepmin)>option.goldensearch_deltmax*(abs(deltmax)) && stepmax > eps
    stepmedr = stepmin+(stepmax-stepmin)/gold;
    stepmedl = stepmin+(stepmedr-stepmin)/gold;
    
    [costmedr,Hstarr] = localCostSimpleMKKM(KH,stepmedr,desc,NN,SigmaNew,numclass);
    [costmedl,Hstarl] = localCostSimpleMKKM(KH,stepmedl,desc,NN,SigmaNew,numclass);     
            
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [val,coord] = min(Cost);
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
            Hstar = Hstarl;
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
            Hstar = Hstarr;
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
            Hstar = Hstarl;
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
            Hstar = Hstarr;
    end
end
%---------------------------------
% Final Updates
%---------------------------------
CostNew = Cost(coord);
step = Step(coord);
% Sigma update
if CostNew < CostOld
    SigmaNew = SigmaNew + step * desc;   
end
Sigma = SigmaNew;