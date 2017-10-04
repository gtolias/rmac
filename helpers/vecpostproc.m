% suite of descriptor post-processing operations
%
% Authors: A. Bursuc, G. Tolias, H. Jegou. 2015. 

function x = vecpostproc(x, a) 
    if ~exist('a'), a = 1; end
    x = replacenan (vecs_normalize (powerlaw (x, a)));

% apply powerlaw
function x = powerlaw (x, a)
if a == 1, return; end
x = sign (x) .* abs(x)  .^ a;

% replace all nan values in a matrix (with zero)
function y = replacenan (x, v)
if ~exist ('v')
  v = 0;
end
y = x;
y(isnan(x)) = v;

% l2 normalization
function X = vecs_normalize(X)
    l = sqrt(sum(X.^2));
    X = bsxfun(@rdivide,X,l);
    X = replacenan(X);