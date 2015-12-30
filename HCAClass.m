function [ output_args ] = HCAClass(data, numb_classes)
%HCAClass Classification pipeline for clustered data
%   input:  - data (vector of length N containing the projected data)
%           - numb_classes (# of classes to target)

% Run initial clustering with PCA
if size(data, 1) < size(data, 2)
    pdat = pca(data);
elseif size(data, 1) > size(data, 2)
    pdat = pca(data');
end

% Label the data
if size(pdat, 1) < size(pdat, 2)
    pdat_labels = LSC(pdat', numb_classes);
elseif size(pdat, 1) > size(pdat, 2)
    pdat_labels = LSC(pdat, numb_classes);
end





end

