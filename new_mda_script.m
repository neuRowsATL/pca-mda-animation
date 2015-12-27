clear; close all; 

rand('state', 1)

to_plot = 1; 

% Get data
new_pca_script;

pdat = projected_data';
pdat_labels = zeros(1, size(pdat, 2));
nr_classes = size(pdat, 2);
class_1 = 0.35;
class_2 = 0.7;
class_3 = 1.0;
for pdat_row=1:size(pdat, 1)
    if pdat_row/length(pdat) <= class_1
        pdat_row/length(pdat);
        pdat_labels(pdat_row) = 1;
    elseif class_2 < pdat_row/length(pdat)
        pdat_row/length(pdat);
        pdat_labels(pdat_row) = 3;
    end
end
pdat_labels(pdat_labels == 0) = 2;

% compute the means and stds of all classes in the N-dim space
nr_features = length(pdat);
nvar = 3;
Means = zeros(nr_classes, nr_features);
Stds = zeros(nr_classes, nr_features);

for ii = 1:nr_classes
    Means(ii, :) = mean(pdat(:, ii));
    Stds(ii, :) =   std(pdat(:, ii));
end

for ii = 1:size(Means, 2)
    for jj = 1:nr_classes
        if(abs(Means(jj, ii) - Means(1, ii)) < 0)
            Means(jj, ii) =  Means(1, ii);
        end
    end
end

weights = ones(1, nr_classes);

% Sw_0 = cov(projected_data - Means(pdat_labels, :));

Sb_exp = zeros(nvar, nvar);
for ii = 1:nr_classes
    Sb_exp(:, :, ii) = weights(ii)*Means(ii, :)'*Means(ii, :);
end

Sw_exp = zeros(nvar, nvar, nr_classes );
for ii = 1:nr_classes
    ii
    size(pdat(find(pdat_labels == ii), :) )
    size(Means(pdat_labels(find(pdat_labels == ii)), :))
    Sw_exp(:, :, ii) = cov(pdat(find(pdat_labels == ii), :) - Means(pdat_labels(find(pdat_labels== ii)), :));
end
Sw_exp_0 = Sw_exp;



for ii = 1:nr_classes
    [v2, d2] = eig(Sw_exp(:, :, ii));
    % compute errors
    errors = zeros(nvar, nr_classes);
    stds_1 = zeros(nvar, nr_classes);
    for jj = 1:nvar
        for kk = 1:nr_classes
            p1 = find(pdat_labels == kk);
            errors(jj, kk) = sum(((pdat(p1, :) - repmat(Means(ii, :), length(p1), 1))*v2(:, jj)).^2);
            stds_1(jj, kk) = std(pdat(p1, :)*v2(:, jj));
        end
    end

    %     [p1, p2] = min(errors(1:nvar - nr_classes, setdiff(1:nr_classes, ii))');
    %     p1 = [p1 errors(nvar - nr_classes + 1:end, ii)'];

    d3 = diag(sum(errors')/sum(sum(errors'))*sum(diag(d2)));
    Sw_exp(:, :, ii) = v2*d3*v2';
    %     Sw_exp(:, :, ii) = eye(nvar);
    %     Sw_exp(:, :, ii) = S_corr(:, :, ii);
end


Sb_0 = zeros(nvar, nvar);
for ii = 1:nr_classes 
    p1 = rand(size(Sb_0)) - 1/2; p1 = p1 + p1';
    Sb_0 = Sb_0 + Sb_exp(:, :, ii);
end

Sw_0 = zeros(nvar, nvar);
for ii = 1:nr_classes; Sw_0 = Sw_0 + Sw_exp(:, :, ii) ; end


% Sw_0 = zeros(nvar, nvar);
% for ii = 1:nr_classes + 1; Sw_0 = Sw_0 + weights(ii)*S_corr(:, :, ii); end
%
% mean_magnitude_responses = mean([repmat(magnitude_responses(1, :), 4, 1); magnitude_responses(2:5, :)]);
% Sb_0 = zeros(nvar, nvar);
% for ii = 1:nr_classes + 1; Sb_0 = Sb_0 + weights(ii)*(magnitude_responses(ii, :) - mean_magnitude_responses)'*(magnitude_responses(ii, :) - mean_magnitude_responses); end

% lambda_1 = 0.9961;
lambda_1 = 0.;
% lambda_1 = 1;
lambda_2 = 0.;


Sw = (1 - lambda_1)*Sw_0 + lambda_1*eye(size(Sw_0));
% Sb = (1 - lambda_2)*Sb_0 + lambda_2*eye(size(Sb_0));

Sb = (1 - lambda_2)*Sb_0 + lambda_2*ones(size(Sb_0));


% Sw = 0*Sw;
% for ii = 1:nr_classes + 1
%     Sw = Sw + length(find(pdat_labels == ii))*S_corr(:, :, ii);
% end


[v d] = eig(inv(Sw)*Sb);

% sort the eigenvalues in order of their relevance
[Sorted Order] = sort(diag(d));

% relevant_eigenvalues = Sorted(end:-1:end - 4)';

Order2 = flipud(Order);
% Order2 = Order;

% Disc contains the weighting factors for the projection
Disc = v(:, Order2);

Disc = Disc(:, 1:5);

% compute the projection in the low-dimensional space
y_train = pdat * Disc;
if(nr_features > nr_classes)
    y_train = y_train(:, 1:nr_classes);
end

y_test = test_data * Disc;
if(nr_features > nr_classes)
    y_test = y_test(:, 1:nr_classes);
end

% keep the projection of data points
y_train_set = y_train;
y_test_set = y_test;



if(to_plot == 1)
%     display_clusters_7_points;
    %     display_clusters_7_points_ver2;
    ClusterVis(pdat, pdat_labels)
end


