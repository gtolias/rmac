%
% Authored by G. Tolias, 2015. 
%
function x = MAC_act(x)

  if ~max(size(x, 1), size(x, 2))
	x = zeros(size(x, 3), 1, class(x));
	return;
  end

  x = reshape(max(max(x, [], 1), [], 2), [size(x,3) 1]);