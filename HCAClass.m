function [pdat, labels] = HCAClass(data, numb_classes)
%HCAClass Classification pipeline for clustered data
%   input:  - data (vector of length N containing the projected data)
%           - numb_classes (# of classes to target)

% Run initial clustering with PCA
[eigenvectors1, ~] = eig(cov(data'));
pdat = eigenvectors1(:, end - 2:end)'*data;

% Label the data
labels = kmeans(pdat', numb_classes);

end

