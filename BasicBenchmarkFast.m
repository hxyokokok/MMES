% @Author: hxy
% @Date:   2017-12-30 10:26:19
% @Last Modified by:   Xiaoyu He
% @Last Modified time: 2020-11-02 14:56:10

function fitfun = BasicBenchmarkFast(FuncId,N)
if FuncId == 2 || FuncId == 7 % ill-conditioned felli
  coef_ = 1e6.^((0:N-1)/(N-1));
elseif FuncId == 6 || FuncId == 11
  coef_ = 2+4*(0:N-1)'/(N-1);
end
% x is a set of column vectors
if FuncId >= 7
  [RotM,~] = qr(randn(N));
end

switch FuncId
	case 1
		fitfun = @(x) fsphere(x);
	case 2
		fitfun = @(x) felli(x,coef_);
	case 3
		fitfun = @(x) frosen(x);
	case 4
		fitfun = @(x) ftablet(x);
	case 5
		fitfun = @(x) fcigar(x);
	case 6
		fitfun = @(x) fdiffpow(x,coef_);
	case 7
		fitfun = @(x) felli(RotM*x,coef_);
	case 8
		fitfun = @(x) frosen(RotM*x);
	case 9
		fitfun = @(x) ftablet(RotM*x);
	case 10
		fitfun = @(x) fcigar(RotM*x);
	case 11
		fitfun = @(x) fdiffpow(RotM*x,coef_);
end

function f=fsphere(x)
	f = sum(x.^2,1);

function f=felli(x,coef_)
	f = coef_ * x.^2;

function f=frosen(x)
	f = 1e2*sum((x(1:end-1,:).^2 - x(2:end,:)).^2,1) + sum((x(1:end-1,:)-1).^2,1);

function f=ftablet(x)
	f = 1e6*x(1,:).^2 + sum(x(2:end,:).^2, 1);

function f=fcigar(x)
	f = x(1,:).^2 + 1e6*sum(x(2:end,:).^2,1);

function f=fdiffpow(x,coef_)
	f = sum(abs(x).^coef_, 1);
