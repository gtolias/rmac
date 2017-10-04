function im = crop_qim(imfn, bbx)

	im = imread(imfn);
	bbx = uint32(max(bbx + 1,  1));
	im = im(bbx(2):min(bbx(4),size(im,1)), bbx(1):min(bbx(3),size(im,2)), :);
