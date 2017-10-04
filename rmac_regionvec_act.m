%
% Authored by G. Tolias, 2015. 
%
function vecs = rmac_regionvec_act(X, L)

ovr = 0.4; % desired overlap of neighboring regions
steps = [2 3 4 5 6 7]; % possible regions for the long dimension

W = size(X, 2);
H = size(X, 1);

w = min([W H]);
w2 = floor(w/2 -1);

b = (max(H, W)-w)./(steps-1);
[~, idx] = min(abs(((w.^2 - w.*b)./w.^2)-ovr)); % steps(idx) regions for long dimension

% region overplus per dimension
Wd = 0;
Hd = 0;
if H < W  
  Wd = idx;
elseif H > W
  Hd = idx;
end

vecs = [];

for l = 1:L

  wl = floor(2*w./(l+1));
  wl2 = floor(wl/2 - 1);

  b = (W-wl)./(l+Wd-1);  
  if isnan(b), b = 0; end % for the first level
  cenW = floor(wl2 + [0:l-1+Wd]*b) -wl2; % center coordinates
  b = (H-wl)./(l+Hd-1);
  if isnan(b), b = 0; end % for the first level
  cenH = floor(wl2 + [0:l-1+Hd]*b) - wl2; % center coordinates

  for i_ = cenH
    for j_ = cenW
      R = X(i_+[1:wl],j_+[1:wl],:);
      if ~min(size(R))
        continue;
      end
      x = mac_act(R); % get mac per region
      vecs = [vecs, x];
    end
  end

end

