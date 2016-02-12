% rand('state', 1);

nr_neurons = 20; 
nvar = nr_neurons; 

nr_repetitions = 7; 
% # of the replicates for each class

nr_classes = 5;
% five classes

mag_noise = ones(1, nr_classes);
mag_separations = 0.5*[1 2 3 4 5];
% mag_separations = mag_separations  - mean(mag_separations);

min_variation = 1;

S_corr = zeros(nvar, nvar, nr_classes);

magnitude_responses = zeros(nr_classes, nvar);

% create covariance matrices and means for 5 classes 
for ii = 1:nr_classes
    p1 = (rand(1, nvar) + min_variation)*mag_separations(ii);
    d = diag(p1);
    

    p2 = 2*rand(nvar, 2*nvar) - 1; 
    p3 = orth(p2);
    % returns an orthonormal basis for the range of p2
        
    S_corr(:, :, ii) = p3*d*p3';
    %     magnitude_responses(ii, :) = mag_separations(ii)*(2*rand(1, nvar) - 1);
    magnitude_responses(ii, :) = mag_separations(ii)*(2*rand(1, nvar) - 1);
   
end
% S_corr  correlation matrix for each group ii from 1 to 5

magnitude_responses = magnitude_responses - ... 
    repmat(mean(magnitude_responses), size(magnitude_responses, 1), 1);

experiment_id = []; 
for ii = 1:nr_classes 
    experiment_id = [experiment_id ii*ones(1, nr_repetitions)];
end

invalid_experiments = [];

test_set = [];
% number of the data which are chosen to be test data

for ii = 1:length(unique(experiment_id))
    p1 = find(experiment_id == ii);
    p1 = setdiff(p1, invalid_experiments);
    % setdiff returns factors in p1 but not in invalid_experiments
    test_set = [test_set p1(floor(rand*nr_repetitions) + 1)];
    test_set = [test_set p1(nr_repetitions + 1:end)];
end


% define the training
training_set = setdiff(1:length(experiment_id), [test_set]);

std_threshold = 0;
nvar0 = nvar;
% to_plot = 1;
experiment_id0 = experiment_id;
select_projection = [1 2 3];


% p1_knockout_general = [];

% in case we want to merge categories, such as air 1 and 2

% g defines the class membership
g = [experiment_id];

g_training_set = g(training_set);
g_test_set = g(test_set);

p1 = 1:nvar;
p1_eliminate_neurons = [];


selected_features = setdiff(p1, p1_eliminate_neurons);
% selected_features = [all_p1_clique_neurons];
% selected_features = [all_p1_clique_neurons, all_p1_clique_neurons + nvar];

% selected_features = p1;
nvar = nr_neurons;
selected_neurons = unique(mod(selected_features - 1, nvar) + 1);

nr_features = length(selected_features);
dim_projection = min(nr_features, nr_classes);

% magnitude_responses = [ones(1, nvar); 2*ones(1, nvar); 3*ones(1, nvar); 4*ones(1, nvar); 5*ones(1, nvar)];
% std_error = ones(size(std_error));


X = zeros(length(g), nvar); 
%what is X????
for ii = 1:nr_classes
    p1 = find(g == ii);
    X(p1, :) = repmat(magnitude_responses(ii, :), length(p1), 1);
    % construct length(p1)*1 matrix with magnitude_response
end

counter = 0; 
for ii = 1:nr_classes
    p1 = find(g == ii);
    for jj = 1:length(p1)
        counter = counter + 1;
        %         p2 = normrnd(0, magnitude_std_error, 1, nvar);
        p2 = mvnrnd(zeros(1, nvar), S_corr(:, :, ii), 1);
        %multivariate normal dist zeros mean and S as correlation matrix
       X(counter, :) = X(counter, :)  + p2;
    end
end

% X = X - repmat(mean(X), size(X, 1), 1);


% ii = 1;     p1 = find(g == ii);
% for jj = 1:nvar
%     X(p1, jj) = normrnd(0, magnitude_std_error, 1, nr_classes*nr_repetitions)' ;
% end


training_data = X(training_set, selected_features);
test_data = X(test_set, selected_features);

training_data_0 = X(training_set, selected_features);
test_data_0 = X(test_set, selected_features);

% test_data = test_data - repmat(mean(training_data), size(test_data, 1), 1);
% training_data = training_data - repmat(mean(training_data), size(training_data, 1), 1);


% compute the means and stds of all classes in the N-dim space
Means = zeros(nr_classes, nr_features);
Stds = zeros(nr_classes, nr_features);

for ii = 1:nr_classes
    Means(ii, :) = mean(training_data(find(g_training_set == ii), :), 1);
    Stds(ii, :) =   std(training_data(find(g_training_set == ii), :), 1);
end

for ii = 1:size(Means, 2)
    for jj = 1:nr_classes
        if(abs(Means(jj, ii) - Means(1, ii)) < std_threshold)
            Means(jj, ii) =  Means(1, ii);
        end
    end
end


Means_th = magnitude_responses;

% Substract the mean and divide by the average variance
Z = training_data;
Z_test = test_data;



