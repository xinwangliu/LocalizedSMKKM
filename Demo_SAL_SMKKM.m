path = '*';
prefix = './';
addpath(genpath(prefix));

dataName = 'Reuters';
load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
numclass = length(unique(Y));
Y(Y<1)=numclass;
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-10;   % numerical precision weights below this value
% are set to zero
%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=500;             % maximal number of iteration
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one

options.miniter=0;                 % minimal number of iterations
options.threshold = 1e-4;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
options.goldensearchmax=1e-3;
options.seuildiffsigma=5e-3;        % stopping criterion for weight variation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



KH = kcenter(KH);
KH = knorm(KH);


%% SAL-SMKKM
tau = 0.45;
tic;
[H_normalized,Sigma,Beta,objs] = SampleAdaptiveLocalSimpleMKKM(KH,tau,numclass,options);
[res_allmean,res_allstd] = myNMIACCV2(H_normalized,Y,numclass);     
timecost(it)=toc;
            




