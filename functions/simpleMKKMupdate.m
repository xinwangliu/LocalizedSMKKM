function [Sigma,Hstar,CostNew] = simpleMKKMupdate(KH,Sigma,GradNew,CostNew,HstarNew,numclass,option)
% Updated at 2021.12.29 23.00 By Zhangyi
%Author Xinwang Liu
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
gold = (sqrt(5)+1)/2 ;
SigmaInit = Sigma ;
SigmaNew  = SigmaInit;

NormGrad = GradNew'*GradNew;
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
Hstar = HstarNew;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------
switch option.firstbasevariable
    case 'first'
        [val,coord] = max(SigmaNew) ;
        %[val,coord] = max(trSTp) ;
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
% GradNew = GradNew - (trSTp/trSTp(coord))*GradNew(coord) ;
% desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) ) ;
% desc(coord) = - sum( trSTp.* desc )/trSTp(coord);  % NB:  GradNew(coord) = 0
GradNew = GradNew - GradNew(coord);
desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) );
desc(coord) = - sum(desc);  % NB:  GradNew(coord) = 0

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
% if stepmax > 1
%     stepmax = 0.1;
% else
%     stepmax = 0.1*stepmax;
% end
% deltmax = stepmax;
%-----------------------------------------------------
%  Projected gradient
%-----------------------------------------------------

[costmax,~] = costSimpleMKKM(KH,stepmax,desc,SigmaInit,numclass,option);
% while costmax<costmin
%     [costmax,Hstar] = costSimpleMKKM(KH,stepmax,desc,SigmaNew,numclass);
%
%     if costmax<costmin
%         costmin = costmax;
%         SigmaNew  = SigmaNew + stepmax * desc;
%     %-------------------------------
%     % Numerical cleaning
%     %-------------------------------
%     SigmaNew(find(abs(SigmaNew<option.numericalprecision)))=0;
%     SigmaNew=SigmaNew/sum(SigmaNew);
%         % SigmaNew  =SigmaP;
%         % project descent direction in the new admissible cone
%         % keep the same direction of descent while cost decrease
%         %desc = desc .* ( (SigmaNew>0) | (desc>0) ) ;
%         desc = desc .* ( (SigmaNew>option.numericalprecision)|(desc>0));
%         desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));
%         ind = find(desc<0);
%         if ~isempty(ind)
%             stepmax = min(-(SigmaNew(ind))./desc(ind));
%             deltmax = stepmax;
% %             if stepmax > 1
% %                 stepmax = 0.1;
% %             else
% %                 stepmax = 0.1*stepmax;
% %             end
% %             deltmax = stepmax;
%             costmax = 0;
%         else
%             stepmax = 0;
%             deltmax = 0;
%         end
%     end
% end

% stepmax = stepmax/gold;
% deltmax = stepmax;
%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------
Step = [stepmin stepmax];
Cost = [costmin costmax];
% [val,coord] = min(Cost);
% optimization of stepsize by golden search


% if deltmax > 10; deltmax = 1; end %% 2022.01.06
coord = 0;
while (stepmax-stepmin)>option.goldensearch_deltmax*(abs(deltmax)) && stepmax > eps
    
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~] = costSimpleMKKM(KH,stepmedr,desc,SigmaInit,numclass,option);
            [costmedl,~] = costSimpleMKKM(KH,stepmedl,desc,SigmaInit,numclass,option);
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            costmedr = costmedl;
            
            [costmedl,~] = costSimpleMKKM(KH,stepmedl,desc,SigmaInit,numclass,option);
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            costmedl = costmedr;
            
            [costmedr,~] = costSimpleMKKM(KH,stepmedr,desc,SigmaInit,numclass,option);
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~] = costSimpleMKKM(KH,stepmedr,desc,SigmaInit,numclass,option);
            [costmedl,~] = costSimpleMKKM(KH,stepmedl,desc,SigmaInit,numclass,option);
        otherwise
            stepmedr = stepmin+(stepmax-stepmin)/gold;
            stepmedl = stepmin+(stepmedr-stepmin)/gold;
            [costmedr,~] = costSimpleMKKM(KH,stepmedr,desc,SigmaInit,numclass,option);
            [costmedl,~] = costSimpleMKKM(KH,stepmedl,desc,SigmaInit,numclass,option);
            %init
    end
   
        Step = [stepmin stepmedl stepmedr stepmax];
        Cost = [costmin costmedl costmedr costmax];
        [~,coord] = min(Cost);
    %
    %
    %     stepmedl = stepmin+(stepmedr-stepmin)/gold;
    %
    %     [costmedr,Hstarr] = costSimpleMKKM(KH,stepmedr,desc,SigmaNew,numclass);
    %     [costmedl,Hstarl] = costSimpleMKKM(KH,stepmedl,desc,SigmaNew,numclass);
    %
    %     Step = [stepmin stepmedl stepmedr stepmax];
    %     Cost = [costmin costmedl costmedr costmax];
    %     [val,coord] = min(Cost);
    %     switch coord
    %         case 1
    %             stepmax = stepmedl;
    %             costmax = costmedl;
    %             Hstar = Hstarl;
    %         case 2
    %             stepmax = stepmedr;
    %             costmax = costmedr;
    %             Hstar = Hstarr;
    %         case 3
    %             stepmin = stepmedl;
    %             costmin = costmedl;
    %             Hstar = Hstarl;
    %         case 4
    %             stepmin = stepmedr;
    %             costmin = costmedr;
    %             Hstar = Hstarr;
    %     end
end
%---------------------------------
% Final Updates
%---------------------------------
[~,coord] = min(Cost);
CostNew = Cost(coord);
step = Step(coord);
% Sigma update
if CostNew < CostOld
    
    [CostNew,Hstar] = costSimpleMKKM(KH,step,desc,SigmaInit,numclass,option);
    Sigma = SigmaNew + step * desc;
    Sigma(Sigma<option.numericalprecision)=0;
    Sigma=Sigma/sum(Sigma);
else
    Hstar = HstarNew;
    Sigma = SigmaInit;
    CostNew = CostOld;
end