% @Author: hxy
% @Date:   2017-12-30 10:25:58
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2020-11-02 14:53:00

s = RandStream('shr3cong','Seed','shuffle');
RandStream.setGlobalStream(s);
funNo = 4;
D = 1000;
algName = 'MMES';

lb = -5; ub = 5;

opt = [];
x0 = lb + rand(D,1) * (ub-lb);
sigma0 = (ub-lb)*0.3;
opt.stopfitness = 1e-8;
opt.stopsigma = 1e-20;
opt.maxFEs = 2e7;
opt.verbose = 1;
opt.stoptime = inf;
opt.ms = 4;
opt.enable_acceleration = 0;

fobj = BasicBenchmarkFast(funNo,D);
tic
out = MMES(fobj, x0, sigma0, opt);
toc



