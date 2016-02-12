load('fdat.txt')
load('pdat_labels.txt')

nr_classes = length(unique(pdat_labels));

weights = zeros(1, nr_classes);

for ii = 1:nr_classes 
    weights(ii) = length(find(pdat_labels == ii));
end

all_data = (fdat - repmat(mean(fdat')', 1, size(fdat, 2)))./repmat(std(fdat')', 1, size(fdat, 2));
all_data = all_data';

experiment_id = pdat_labels;

nr_neurons = size(all_data, 2); 
nvar = nr_neurons; 

nr_test_points_per_class = 1; 
% # of the replicates for each class


test_set = [];
% number of the data which are chosen to be test data
invalid_experiments = [];

for ii = 1:length(unique(experiment_id))
    p1 = find(experiment_id == ii);
    p1 = setdiff(p1, invalid_experiments);
    % setdiff returns factors in p1 but not in invalid_experiments
    p2 = randperm(length(p1));
    test_set = [test_set p1(p2(1:nr_test_points_per_class))];
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




training_data = all_data(training_set, selected_features);
test_data = all_data(test_set, selected_features);


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




