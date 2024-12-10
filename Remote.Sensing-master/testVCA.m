
load('your_input_path.mat', 'truth')
truth = double(truth) / double(max(truth(:)))

% band1,band2 and band3 are the selected pseudo-RGB color channels.
% rgb(:,:,1) = truth(:,:,band1)
% rgb(:,:,2) = truth(:,:,band2)
% rgb(:,:,3) = truth(:,:,band3)
% imshow(rgb)

[h, w, c] = size(truth)
bandNum = c;
endNum = 12;%12 for matiwan and houston;9 for pavia and chikusei
I2 = reshape(truth, h*w, bandNum)

% run VCA
HVca = hyperVca(I2', endNum);
save('you_output_path.mat', 'HVca', 'truth')


