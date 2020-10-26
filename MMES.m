% @Author: hxy
% @Date:   2018-05-26 20:39:01
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2020-08-23 10:01:52

% Gaussian Mixture Model based ES 

function out = MMES(fobj, dim, opt)
xmean = opt.x0;
sigma_ = opt.sigma0;
stopfitness = opt.stopfitness;
maxFEs = opt.maxFEs;
stopsigma = opt.stopsigma;
stoptime = opt.stoptime;

%% parameters
lambda = 4+floor(3*log(dim));     
mu = floor(lambda/2);
weights = log(mu+1/2)-log(1:mu)';
weights = weights/sum(weights);
mueff = 1/sum(weights.^2);

% number of search vectors 
if ~isfield(opt,'m') m = 2*ceil(sqrt(dim)); else m = opt.m; end
% learning rate for evolution path
if ~isfield(opt,'cc') cc = 0.4; else cc = opt.cc; end
% learning rate for covariance matrix, seems to be set linear inverse to m
if ~isfield(opt,'c1') c1 = 3.8; else c1 = opt.c1; end % can be further investigated
% mixing strength, the distribution converges to Gaussian when %ms% approaches infty, larger the better (but slower) ...
% increase from 10 to 50 will save 3% FEs but increase 50% runtime 
if ~isfield(opt,'ms') ms = 4; else ms = opt.ms; end
% learning rate for step size adaptation
if ~isfield(opt,'cs') cs = 0.3; else cs = opt.cs; end 
% target significance level
if ~isfield(opt,'qtarget') qtarget = 0.05; else qtarget = opt.qtarget; end 
% q*=0.05, cs=0.3 are from SDA-ES
% q*=0.1, cs=0.25 seem to be slightly worse... ....
output_model = isfield(opt,'output_model') && opt.output_model;

c1 = c1/dim;  
cc = cc/sqrt(dim);
T = ceil(1/cc);
% initialization
fbest = fobj(xmean);
prevfits = fbest*ones(1,lambda); 
arX = zeros(dim,lambda);
FEs = 1; 
gen = 1;
% search vectors and evolution path 
PCs = randn(dim,m)*1e-6;
pc = zeros(dim,1);
% generation record and index record
itrs = 1 : m;
pcidx = 1 : m;
l1 = ceil(lambda/2);
% learning rate for each component model
cgamma = 1 - (1-c1)^m;
% accumulated success measurement
scum = 0;  
% for recording
recordGap = 100;
record = zeros(ceil(maxFEs/lambda/recordGap+2e3/lambda),2);
recordIter = 1;

if opt.verbose > 0
    fprintf('MMES: m = %d, ms = %d, c1 = %g/d, cc = %g/sqrt(d), q* = %g, cs = %g\n',...
        m, ms, c1*dim, cc*sqrt(dim), qtarget, cs);
    if output_model
        fprintf('warning: ready to output model.\n');
    end
end

inTic = tic;
while fbest > stopfitness && FEs < maxFEs && sigma_ > stopsigma && toc(inTic) <= stoptime
    %% sampling
    try
        rndk_ = m-mod(geornd(c1,l1,ms),m);
        rndk = pcidx(rndk_);
    catch ME
        rndk_ = m-mod(geornd(c1,l1,ms),m);
        rndk = pcidx(rndk_);
    end
    %% for measuring execution time ... generally slow
%     arY_ = zeros(dim,l1);
%     for i = 1 : ms
%         arY_ = arY_ + PCs(:,rndk(:,i)) .* randn(1,l1);
%     end
    %% for competition ... fast for small mixing strength
    arY_ = MMES_mixing(PCs,rndk,randn(1,l1*ms));

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
        % terminate if a stagnation is detected
        if isfield(opt,'stagnation') && FEs > opt.stagnation.FEsOffset
            ptr = floor(recordIter - opt.stagnation.genGap / recordGap);
            if ptr > 0 && record(ptr,2) - fbest < opt.stagnation.tolObj
                break;
            end
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
% itrs

if output_model
    out.model.mean = xmean;
    out.model.Q = PCs(:,pcidx);
    out.model.sigma = sigma_;
end