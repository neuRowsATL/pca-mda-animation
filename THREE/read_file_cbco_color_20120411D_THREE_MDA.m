clear; close all;

cbco_data = [];

for ii = 1:34
    file1 = ['20120411D_CBCO-' num2str(ii) '.txt'];
    
    fid = fopen(file1);
    
    c1 = textscan(fid, '%s %s %s \n');
    c2 = textscan(fid, '%s %s \n');
    
    c = textscan(fid, '%f', 'Delimiter', '\n', 'CollectOutput', true);
    
    data = c{1}; data = data(2:end)';
    
    cbco_data = [cbco_data; [data; (ii)*ones(size(data))]' ];
end

cbco_data = cbco_data';

firing_times = [cbco_data];

nvar = max(firing_times(2, :));

nr_points = 10e2 - 1;
t_linspace = linspace(min(firing_times(1, :)), max(firing_times(1, :)), nr_points);
delta_t = t_linspace(2) - t_linspace(1);
t_linspace = [t_linspace(1) - delta_t, t_linspace, t_linspace(end) + delta_t];
nr_points = nr_points + 1;

frequency_responses = zeros(nvar, nr_points);
for ii = 1:nvar
    for jj = 1:nr_points
        p1 = find(firing_times(2, :) == ii);
        frequency_responses(ii, jj) = sum(find(firing_times(1, p1) > t_linspace(jj) & firing_times(1, p1) < t_linspace(jj + 1)))/delta_t;
    end
end

frequency_responses = (frequency_responses - repmat(mean(frequency_responses')', 1, size(frequency_responses, 2)))./ ...
    repmat(std(frequency_responses')', 1, size(frequency_responses, 2));

frequency_responses = (1 + tanh(frequency_responses))/2;
%frequency_responses = log(frequency_responses);

[eigenvectors1, eigenvalues1] = eig(cov(frequency_responses'));
projected_data = eigenvectors1(:, end - 2:end)'*frequency_responses;

% projected_data = log(projected_data);
% figure; plot3(projected_data(1, :), projected_data(2, :), projected_data(3, :), '.');

summed_frequencies = sum(frequency_responses);
min_summed_frequencies = min(summed_frequencies);
max_summed_frequencies = max(summed_frequencies);


fid_CL = fopen('CL.txt');

cl_line_one = textscan(fid_CL, '%s %s %s \n');
cl_line_two = textscan(fid_CL, '%s %s\t     %s\n');

%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
cl_data = textscan(fid_CL, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

CL = cl_data{1};


fid_dec_sine = fopen('low_sine.txt');

dec_sine_line_one = textscan(fid_dec_sine, '%s %s %s \n');
dec_sine_line_two = textscan(fid_dec_sine, '%s %s\t     %s\n');

dec_sine_data = textscan(fid_dec_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

dec_sine = dec_sine_data{1};

fid_inf_sine = fopen('inf_sine.txt');

inf_sine_line_one = textscan(fid_inf_sine, '%s %s %s \n');
inf_sine_line_two = textscan(fid_inf_sine, '%s %s\t     %s\n');

inf_sine_data = textscan(fid_inf_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

inf_sine = inf_sine_data{1};


fid_inc_sine = fopen('top_sine.txt');

inc_sine_line_one = textscan(fid_inc_sine, '%s %s %s \n');
inc_sine_line_two = textscan(fid_inc_sine, '%s %s\t     %s\n');

inc_sine_data = textscan(fid_inc_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

inc_sine = inc_sine_data{1};


fid_no_sim = fopen('no_sim.txt');

no_sim_line_one = textscan(fid_no_sim, '%s %s %s \n');
no_sim_line_two = textscan(fid_no_sim, '%s %s\t     %s\n');

no_sim_data = textscan(fid_no_sim, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

no_sim = no_sim_data{1};

fid_tugs_ol = fopen('tugs_ol.txt');

tugs_ol_line_one = textscan(fid_tugs_ol, '%s %s %s \n');
tugs_ol_line_two = textscan(fid_tugs_ol, '%s %s\t     %s\n');

tugs_ol_data = textscan(fid_tugs_ol, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

tugs_ol = tugs_ol_data{1};

end_inc_sine = inc_sine(:,1)+inc_sine(:,2);
end_dec_sine = dec_sine(:,1)+dec_sine(:,2);
end_CL = CL(:,1)+CL(:,2);
end_no_sim = no_sim(:,1)+no_sim(:,2);

linspace_classes = zeros(size(t_linspace));

for qq = 1:size(inc_sine, 1)
    x_inc = t_linspace >= inc_sine(qq,1);
    y_inc = t_linspace <= end_inc_sine(qq,1);
    linspace_classes(x_inc & y_inc)=1;
    %         t_linspace((t_linspace > inc_sine(qq,1)) & (t_linspace<end_inc_sine(qq,1))) = 1;
    %         t_linspace((t_linspace > dec_sine(qq,1) & t_linspace<end_dec_sine(qq,1)) = 2;
    %         t_linspace((t_linspace > CL(qq,1) & CL<end_CL(qq,1)) = 3;
    %         t_linspace((no_sim > CL(qq,1) & CL<end_no_sim(qq,1)) = 4;
end
for qq = 1:size(dec_sine, 1)
    x_dec = t_linspace >= dec_sine(qq,1);
    y_dec = t_linspace <= end_dec_sine(qq,1);
    linspace_classes(x_dec & y_dec)=2;
end

for qq = 1:size(CL, 1)
    x_CL = t_linspace > CL(qq,1);
    y_CL = t_linspace < end_CL(qq,1);
    linspace_classes(x_CL & y_CL)=3;
end

for qq = 1:size(no_sim, 1)
    x_no_sim = t_linspace > no_sim(qq,1);
    y_no_sim = t_linspace < end_no_sim(qq,1);
    linspace_classes(x_no_sim & y_no_sim)=4;
end





