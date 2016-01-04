function KmeansVis(data, labels, no_classes)
%KmeansVis Create a set of ellipsoids centered at each cluster
%   input:  
%       - data : array of doubles; has been projected (PCA) and labeled (k-means)
%       - labels : array of labels (doubles), corresponding to the data
%   output:
%        - kels : array containing [x, y, z] corresponding to each
%        ellipsoid

if size(data, 1) ~= size(labels, 1)
    data = data';
end

colors = ['r', 'b', 'k', 'm', 'c', 'g', 'y'];
set(gca, 'BoxStyle', 'full', 'Box', 'on');
hold on
for ii=1:no_classes
    data_in_class = data(labels==ii, :);
    
    col1 = data_in_class(:, 1);
    col2 = data_in_class(:, 2);
    col3 = data_in_class(:, 3);
    
    max1 = max(col1);
    max2 = max(col2);
    max3 = max(col3);
    
    min1 = min(col1);
    min2 = min(col2);
    min3 = min(col3);
    
    c1 = median(col1);
    c2 = median(col2);
    c3 = median(col3);
    
    [x, y, z] = ellipsoid(c1, c2, c3, std(col1), std(col2), std(col3));
    surf(x, y, z, 'FaceColor', colors(ii),...
        'EdgeColor', 'none');
    alpha(0.4);
    alphamap('vup');
    view([30 30 15])
end

end

