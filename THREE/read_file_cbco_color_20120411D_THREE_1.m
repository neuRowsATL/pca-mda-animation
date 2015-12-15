clear; close all;

cbco_data = [];

for ii = 1:34
    file1 = ['20120411D_CBCO-' num2str(ii) '.txt'];
    
    fid = fopen(file1);
    
    % c = textscan(fid, ...
    %                 '%*s v1=%f v2=%f v3=%f %*s', ...
    %                 'Delimiter', '\n', ...
    %                 'CollectOutput', true);
    
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
    ii
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

%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
dec_sine_data = textscan(fid_dec_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

dec_sine = dec_sine_data{1};

fid_inf_sine = fopen('inf_sine.txt');

inf_sine_line_one = textscan(fid_inf_sine, '%s %s %s \n');
inf_sine_line_two = textscan(fid_inf_sine, '%s %s\t     %s\n');

%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
inf_sine_data = textscan(fid_inf_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

inf_sine = inf_sine_data{1};


fid_inc_sine = fopen('top_sine.txt');

inc_sine_line_one = textscan(fid_inc_sine, '%s %s %s \n');
inc_sine_line_two = textscan(fid_inc_sine, '%s %s\t     %s\n');


%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
inc_sine_data = textscan(fid_inc_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

inc_sine = inc_sine_data{1};


fid_no_sim = fopen('no_sim.txt');

no_sim_line_one = textscan(fid_no_sim, '%s %s %s \n');
no_sim_line_two = textscan(fid_no_sim, '%s %s\t     %s\n');


%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
no_sim_data = textscan(fid_no_sim, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

no_sim = no_sim_data{1};

fid_tugs_ol = fopen('tugs_ol.txt');

tugs_ol_line_one = textscan(fid_tugs_ol, '%s %s %s \n');
tugs_ol_line_two = textscan(fid_tugs_ol, '%s %s\t     %s\n');


%  ca = textscan(fid, '%f %f', 'Delimiter', '\n', 'CollectOutput', true);
tugs_ol_data = textscan(fid_tugs_ol, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);

tugs_ol = tugs_ol_data{1};


figure; hold on;
for ii = 1:size(projected_data, 2)
    
    % if ((EVENT ON TIME < time_data) & (time_data < EVENT END TIME)) |
    % if ((EVENT ON TIME < time_data) & (time_data < EVENT ON TIME + DURATION)) |
    % if ((row 1 column 1 of data file < time_data) & (time_data < r1c1 + r1c2)) |
    time_data = t_linspace(ii + 1)
    a = find( diff(sign(CL(:,1)-time_data)) == 2);
    b = find( diff(sign(dec_sine(:,1)-time_data)) == 2);
    c = find( diff(sign(inc_sine(:,1)-time_data)) == 2);
    d = find( diff(sign(no_sim(:,1)-time_data)) == 2);
    e = find( diff(sign(tugs_ol(:,1)-time_data)) == 2);
    f = find( diff(sign(inf_sine(:,1)-time_data)) == 2);
    
    if (time_data > CL(a,1)) & (time_data < CL(a,1) + CL(a,2))
        color1 = 'r'
    elseif (time_data > dec_sine(b,1)) & (time_data < dec_sine(b,1) + dec_sine(b,2))
        color1 = 'm'
    elseif (time_data > inc_sine(c,1)) & (time_data < inc_sine(c,1) + inc_sine(c,2))
        color1 = 'k'
    elseif (time_data > inf_sine(f,1)) & (time_data < inf_sine(f,1) + inf_sine(f,2))
        color1 = 'c'
    elseif (time_data > no_sim(d,1)) & (time_data < no_sim(d,1) + no_sim(d,2))
        color1 = 'b'
    elseif (time_data > tugs_ol(e,1)) & (time_data < tugs_ol(e,1) + tugs_ol(e,2))
        color1 = 'y'
    else
        color1 = 'g'
    end
    
    symbol_1 = '*';
    plot3(projected_data(1, ii), projected_data(2, ii), projected_data(3, ii),'Marker', symbol_1, 'Color', color1, ...
        'MarkerSize', 6, 'LineStyle', '-');
    %     if ii > 1
    %         plot3([projected_data(1, ii-1), projected_data(1, ii)], [projected_data(2, ii-1), projected_data(2, ii)], [projected_data(3, ii-1), projected_data(3,ii)], 'Color', color1);
    %     end
end
xlabel('PCA 1', 'FontSize', 18); ylabel('PCA 2', 'FontSize', 18); zlabel('PCA 3', 'FontSize', 18)
% plot3(projected_data(1, :), projected_data(2, :), projected_data(3, :));
% title([experiment_filename ' PCA'], 'FontSize', 18)


view([30, 30])



