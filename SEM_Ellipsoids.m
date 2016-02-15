fdat = load('fdat.txt');
labels = load('pdat_labels.txt');

pdat = pca(fdat, 'NumComponents', 3);

sem = zeros(max(labels), 3);
for lab=1:max(labels)
    sem(lab, :) = std(pdat(labels==lab, :)) / sqrt(numel(labels(labels==lab)));
end

close all;
cla;

xlabel('SE P1');
ylabel('SE P2');
zlabel('SE P3');
title('Standard Error : PCA, CMA')
view([15 30])
hold on;
for lab=1:max(labels)
    color1 = colors(lab);
    cent = mean(pdat(labels==lab, :));
    rad = sem(lab, :);
    ellipsoid(cent(1), cent(2), cent(3), rad(1), rad(2), rad(3));
end
hold off;