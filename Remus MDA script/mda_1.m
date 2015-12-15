clear; close all; 

rand('state', 1)

to_plot = 1; 

create_data;

% Means = Means_th;

weights = ones(1, nr_classes);

% weights = [1 4 4 4 4];

Sw_0 = cov(training_data - Means(g_training_set, :));

Sb_exp = zeros(nvar, nvar);
for ii = 1:nr_classes
    Sb_exp(:, :, ii) = weights(ii)*Means(ii, :)'*Means(ii, :);
end

Sw_exp = zeros(nvar, nvar, nr_classes );
for ii = 1:nr_classes
    Sw_exp(:, :, ii) = cov(training_data(find(g_training_set == ii), :) - Means(g_training_set(find(g_training_set == ii)), :));
end
Sw_exp_0 = Sw_exp;



for ii = 1:nr_classes
    [v2, d2] = eig(Sw_exp(:, :, ii));
    % compute errors
    errors = zeros(nvar, nr_classes);
    stds_1 = zeros(nvar, nr_classes);
    for jj = 1:nvar
        for kk = 1:nr_classes
            p1 = find(g_training_set == kk);
            errors(jj, kk) = sum(((training_data(p1, :) - repmat(Means(ii, :), length(p1), 1))*v2(:, jj)).^2);
            stds_1(jj, kk) = std(training_data(p1, :)*v2(:, jj));
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
%     Sw = Sw + length(find(g_training_set == ii))*S_corr(:, :, ii);
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
    %     display_clusters_7_points_ver2;
end


