function [pdat, labels] = HCAClass(data, numb_classes)
%HCAClass Classification pipeline for clustered data
%   input:  - data (vector of length N containing the projected data)
%           - numb_classes (# of classes to target)

% Run initial clustering with PCA
[eigenvectors1, ~] = eig(cov(data'));
pdat = eigenvectors1(:, end - 2:end)'*data;

% Label the data
if size(pdat, 1) < size(pdat, 2)
    pdat_labels = LSC(pdat', numb_classes);
elseif size(pdat, 1) > size(pdat, 2)
    pdat_labels = LSC(pdat, numb_classes);
end

% SVM classifier
% Function from here: http://www.mathworks.com/matlabcentral/fileexchange/39352-multi-class-svm
% Author: Cody
% training_x = pdat(:, 1:length(pdat)*(2/5))';
% training_y = pdat_labels(1:length(pdat_labels)*(2/5), :);
% test_x = pdat(:, length(pdat)*(2/5):end)';
% results = multisvm(training_x, training_y, test_x);
% second_results = multisvm(test_x, results, training_x);
labels = kmeans(pdat', numb_classes);
end

