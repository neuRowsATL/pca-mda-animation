cla; 

rand('state', 1)

to_plot = 1; 

load_data;

% Means = Means_th;

weights = ones(1, nr_classes);

for ii = 1:nr_classes
    weights(ii) = length(find(experiment_id == ii));
end
% weights = [1 4 4 4 4];


Sb_0 = zeros(nvar, nvar);
Sw_0 = zeros(nvar, nvar);
Sw_exp = zeros(nvar, nvar, nr_classes);
for ii = 1:nr_classes
    Sb_exp(:, :, ii) = weights(ii)*Means(ii, :)'*Means(ii, :);
    Sb_0 = Sb_0 + Sb_exp(:, :, ii);
end

Sw_exp = zeros(nvar, nvar, nr_classes );
for ii = 1:nr_classes
    Sw_exp(:, :, ii) = cov(training_data(find(g_training_set == ii), :) - Means(g_training_set(find(g_training_set == ii)), :));
    Sw_0 = Sw_0 + Sw_exp(:, :, ii);
end



% lambda_1 = 0.9961;
lambda_1 = 0.;
% lambda_1 = 1;
lambda_2 = 0.;


Sw = (1 - lambda_1)*Sw_0 + lambda_1*eye(size(Sw_0));
% Sb = (1 - lambda_2)*Sb_0 + lambda_2*eye(size(Sb_0));

Sb = (1 - lambda_2)*Sb_0 + lambda_2*ones(size(Sb_0));



[v d] = eig(inv(Sw)*Sb);

% sort the eigenvalues in order of their relevance
[Sorted Order] = sort(diag(d));

% relevant_eigenvalues = Sorted(end:-1:end - 4)';

Order2 = flipud(Order);
% Order2 = Order;

% Disc contains the weighting factors for the projection
Disc = v(:, Order2);

Disc = Disc(:, 1:nr_classes);

% compute the projection in the low-dimensional space
y_train = training_data * Disc;
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
    display_clusters_7_points;
% mda_se_ellipsoids;
%     %     display_clusters_7_points_ver2;
end


