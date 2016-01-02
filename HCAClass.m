function [pdat, labels] = HCAClass(data, numb_classes)
%HCAClass Classification pipeline for clustered data
%   input:  - data (vector of length N containing the projected data)
%           - numb_classes (# of classes to target)

% Run initial clustering with PCA
[eigenvectors1, ~] = eig(cov(data'));
pdat = eigenvectors1(:, end - 2:end)'*data;

% Label the data
labels = kmeans(pdat', numb_classes, 'Replicates', 10);

% Experiment showing that 4 clusters is the most likely for this data
% for ii=1:10
% idx5 = kmeans(pdat', ii, 'Display', 'final', 'Replicates', 10, 'MaxIter', 1e4);
% silh5 = silhouette(pdat',idx5);
% disp(ii)
% mean(silh5)
% end
% figure;
% [silh5,h] = silhouette(pdat',idx5);
% h = gca;
% h.Children.EdgeColor = [.8 .8 1];
% xlabel 'Silhouette Value';
% ylabel 'Cluster';
% mean(silh5)
end

