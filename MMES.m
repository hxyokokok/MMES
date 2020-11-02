% @Author: hexy_
% @Date:   2020-11-02 11:42:28
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2020-11-02 14:46:16

% Gaussian Mixture Model based ES 

%% input parameters
% fobj: function handle of the optimization problem
% x0: initial point
% sigma0: initial mutation strength
% opt: parameter option
%	opt.stopfitness / opt.stopsigma / opt.stoptime: stopping criteria, the algorithm stops 
% 		if 1) the best objective function value found is smaller than opt.stopfitness; or
% 		2) the mutation strength is smaller than opt.stopsigma; or 3) the computing time (s)
% 		exceeds opt.stoptime.
% 	opt.verbose: 1=> show information; 0=> run without information display

function out = MMES(fobj, x0, sigma0, opt)
xmean = x0;
sigma_ = sigma0;
% dimensionality
dim = length(x0);

%% enable acceleration
% in the sampling process i provide a mex implementation for acceleration. set opt.acceleration>0 to enable this.
% for comparing computing time please turn off this acceleration.
accel = isfield(opt,'enable_acceleration') && opt.enable_acceleration > 0;

%% stopping criteria
stopfitness = opt.stopfitness;
maxFEs = opt.maxFEs;
stopsigma = opt.stopsigma;
stoptime = opt.stoptime;

%% parameters from the standard ES framework. 
% do NOT change them.
lambda = 4+floor(3*log(dim)); % population size    
mu = floor(lambda/2); % selected solution number 
weights = log(mu+1/2)-log(1:mu)'; % weights
weights = weights/sum(weights);
mueff = 1/sum(weights.^2); % a useful constant; see the tutorial of CMA-ES for its meaning

%% parameters of the mixture model
% number of search vectors; the default setting is recommended by many related works 
if ~isfield(opt,'m') m = 2*ceil(sqrt(dim)); else m = opt.m; end

% learning rate for evolution path: roughly tuned to achieve universal good performance
% note that theoretical studies suggest that its optimal value should be inversely proportional to sqrt(dim), 
% 	so, here, this parameter only denotes the scaling coefficient of the dim^(-0.5) term
if ~isfield(opt,'cc') cc = 0.4; else cc = opt.cc; end

% learning rate for covariance matrix: roughly tuned to achieve universal good performance
% a well accepted heuristic states that its value should be inversely proportional to the dimension,
% 	and thus we only set the scaling coefficient of the 1/dim term
if ~isfield(opt,'c1') c1 = 3.8; else c1 = opt.c1; end 

% mixing strength, the distribution converges to Gaussian when ms approaches infinity, larger the better (but slower) ...
% the setting ms = 4 is strongly recommended and suitable for most cases, please see the supplement for more details
if ~isfield(opt,'ms') ms = 4; else ms = opt.ms; end

% learning rate for step size adaptation
% the setting of this parameter and the next one is from the paper of SDA-ES; 
% 	indeed the paired test adaptation (PTA) differs from the generalized 1/5-th success rule (GSR) of SDA-ES 
% 	only in the measurement of the success.
if ~isfield(opt,'cs') cs = 0.3; else cs = opt.cs; end 
% target significance level
if ~isfield(opt,'qtarget') qtarget = 0.05; else qtarget = opt.qtarget; end 

% scale the learning rate
c1 = c1/dim;  
cc = cc/sqrt(dim);

%% initialization
% best objective found
fbest = fobj(xmean); 
% fitness of the previous population 
prevfits = fbest*ones(1,lambda); 
% decision variables of the current population
arX = zeros(dim,lambda);
% function evaluations
FEs = 1; 
% generation index
gen = 1;
% stored search vectors 
PCs = randn(dim,m)*1e-6;
% current evolution path 
pc = zeros(dim,1);

%% parameters from LM-CMA: MMES uses existing method to maintain the search direction; the idea is from LM-CMA which keeps an fixed interval between consecutive search directions.
% 	LM-CMA maintains two different types of evolution paths while we have used its simplified version proposed in Rm-ES which only stores one type of evolution paths.
% the generation indexes storing when the evolution paths are generated
itrs = 1 : m;
% indexes to the stored evolution paths
pcidx = 1 : m;
% interval of updating the evolution paths
T = ceil(1/cc);

% unless otherwise stated, we use mirror sampling for acceleration. 
% a number of l1 solutions are sampled firstly while the rest lambda-l1 ones are obtained using their mirroring
l1 = ceil(lambda/2);

% learning rate for each component model
cgamma = 1 - (1-c1)^m;
% accumulated success measurement
scum = 0;  

% parameter for recording
recordGap = 100; % the interval of generations for recording
record = zeros(ceil(maxFEs/lambda/recordGap+2e3/lambda),2);
recordIter = 1;

if opt.verbose > 0
    fprintf('MMES: m = %d, ms = %d, c1 = %g/d, cc = %g/sqrt(d), q* = %g, cs = %g\n',...
        m, ms, c1*dim, cc*sqrt(dim), qtarget, cs);
end

inTic = tic;
while fbest > stopfitness && FEs < maxFEs && sigma_ > stopsigma && toc(inTic) <= stoptime
    %% mirror sampling
    % step-1: select linear Gaussian models using a geometric distribution, 
    % 	(I find this sometimes crashes, so I use an exception handling process to re-sample) 
    try
        rndk_ = m-mod(geornd(c1,l1,ms),m);
        rndk = pcidx(rndk_);
    catch ME
        rndk_ = m-mod(geornd(c1,l1,ms),m);
        rndk = pcidx(rndk_);
    end
    % step-2: drawing samples from the linear Gaussian models
    if ~accel % use plain matlab; generally slow
	    arY_ = zeros(dim,l1);
	    for i = 1 : ms
	        arY_ = arY_ + PCs(:,rndk(:,i)) .* randn(1,l1);
	    end
	else %% use mex for acceleration; fast for small mixing strength
	    arY_ = MMES_mixing(PCs,rndk,randn(1,l1*ms));
	end

	% step-3: append isotropic Gaussians
    arY_ = sqrt(cgamma/ms) * arY_ + sqrt(1-cgamma) * randn(dim,l1); % Bottleneck

    arX(:,1:l1) = arY_;
    arX(:,l1+1:end) = -arY_(:,1:lambda-l1);
    arX = xmean + sigma_ * arX ;

    %% evaluate and recombine
    arFitness = fobj(arX); 
    FEs = FEs+lambda;
    [arFitness, arIndex] = sort(arFitness);
    xold = xmean;
    arIndex = arIndex(1:mu);
    xmean = arX(:,arIndex) * weights;  
    z = sqrt(mueff) * (xmean-xold) / sigma_;
    pc = (1-cc) * pc + sqrt(cc*(2-cc))*z;
    
    %% update search directions
    if gen <= m
        PCs(:,gen) = pc;
    else
        [valmin, imin] = min(itrs(pcidx(2:m)) - itrs(pcidx(1:m-1)));
        imin = imin + 1;
        if valmin > T 
            imin = 1;
        end
        pcidx = pcidx([1:imin-1, imin+1:m, imin]);
        itrs(pcidx(m)) = gen;
        PCs(:,pcidx(m)) = pc;
    end
    
    %% pairwise test adaptation
    L = sum(weights(arFitness(1:mu)<prevfits(1:mu)));
    W = (2*L-1)*sqrt(mueff);
    scum = (1-cs) *scum + sqrt(cs*(2-cs))*W;
    sigma_ = sigma_ * exp(normcdf(scum)-1+qtarget); 
    prevfits = arFitness;
    
    %% recording
    fbest = min(fbest, arFitness(1));
    if mod(gen,recordGap) == 1 
        record(recordIter,:) = [FEs fbest];
        if opt.verbose > 0
            fprintf('#%d FEs=%d fit=%g\n', recordIter, FEs, fbest);
        end
        recordIter = recordIter + 1;
    end
    gen = gen + 1;  
end
record(recordIter,:) = [FEs fbest];
if size(record,1) > recordIter
    record(recordIter+1:end,:) = [];
end
out.record = record;
out.bestFitness = fbest;
