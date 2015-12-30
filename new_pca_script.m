%% Get data
clear; close all;
cbco_data = [];

for ii = 1:34
    file1 = ['THREE/20120411D_CBCO-' num2str(ii) '.txt'];
    
    fid = fopen(file1);
    
    
    c1 = textscan(fid, '%s %s %s \n');
    c2 = textscan(fid, '%s %s \n');
    
    c = textscan(fid, '%f', 'Delimiter', '\n', 'CollectOutput', true);
    
    data = c{1}; data = data(2:end)';
    
    cbco_data = [cbco_data; [data; (ii)*ones(size(data))]' ];
end

cbco_data = cbco_data';

%% Initial Clusering
firing_times = cbco_data;

nvar = max(firing_times(2, :));

nr_points = 10e2 - 1;
t_linspace = linspace(min(firing_times(1, :)), max(firing_times(1, :)), nr_points);
delta_t = t_linspace(2) - t_linspace(1);
t_linspace = [t_linspace(1) - delta_t, t_linspace, t_linspace(end) + delta_t];
nr_points = nr_points + 1;

% Find frequency responses
frequency_responses = zeros(nvar, nr_points);
for ii = 1:nvar
    ii;
    for jj = 1:nr_points
        p1 = find(firing_times(2, :) == ii);
        frequency_responses(ii, jj) = sum(find(firing_times(1, p1) > t_linspace(jj) & firing_times(1, p1) < t_linspace(jj + 1)))/delta_t;
    end
end
frequency_responses = (frequency_responses - repmat(mean(frequency_responses')', 1, size(frequency_responses, 2)))./ ...
    repmat(std(frequency_responses')', 1, size(frequency_responses, 2));
frequency_responses = (1 + tanh(frequency_responses))/2;

% Run PCA
opt = statset('ppca');
opt.MaxIter = 100e3;
opt.TolFun = 1e-8;
opt.TolX = 1e-8;
[pdat0, score, ~] = ppca(cov(frequency_responses), 3, 'Options', opt);
no_clust = 5;
pdat = clusterdata(pdat0, 'maxclust', no_clust);
for jj = 1:no_clust
    jj;
    print_clust(:, jj) = [jj; nnz(pdat==jj)];
end
print_clust
%% Classification data
fid_CL = fopen('THREE/CL.txt');
cl_line_one = textscan(fid_CL, '%s %s %s \n');
cl_line_two = textscan(fid_CL, '%s %s\t     %s\n');
cl_data = textscan(fid_CL, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
CL = cl_data{1};

fid_dec_sine = fopen('THREE/low_sine.txt');
dec_sine_line_one = textscan(fid_dec_sine, '%s %s %s \n');
dec_sine_line_two = textscan(fid_dec_sine, '%s %s\t     %s\n');
dec_sine_data = textscan(fid_dec_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
dec_sine = dec_sine_data{1};

fid_inf_sine = fopen('THREE/inf_sine.txt');
inf_sine_line_one = textscan(fid_inf_sine, '%s %s %s \n');
inf_sine_line_two = textscan(fid_inf_sine, '%s %s\t     %s\n');
inf_sine_data = textscan(fid_inf_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
inf_sine = inf_sine_data{1};

fid_inc_sine = fopen('THREE/top_sine.txt');
inc_sine_line_one = textscan(fid_inc_sine, '%s %s %s \n');
inc_sine_line_two = textscan(fid_inc_sine, '%s %s\t     %s\n');
inc_sine_data = textscan(fid_inc_sine, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
inc_sine = inc_sine_data{1};

fid_no_sim = fopen('THREE/no_sim.txt');
no_sim_line_one = textscan(fid_no_sim, '%s %s %s \n');
no_sim_line_two = textscan(fid_no_sim, '%s %s\t     %s\n');
no_sim_data = textscan(fid_no_sim, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
no_sim = no_sim_data{1};

fid_tugs_ol = fopen('THREE/tugs_ol.txt');
tugs_ol_line_one = textscan(fid_tugs_ol, '%s %s %s \n');
tugs_ol_line_two = textscan(fid_tugs_ol, '%s %s\t     %s\n');
tugs_ol_data = textscan(fid_tugs_ol, '%f %f', 'Delimiter', '\t', 'CollectOutput', true);
tugs_ol = tugs_ol_data{1};

%% LSC(), ClusterVis()
p1_dat = pdat(1, :);
p2_dat = pdat(2, :);
p3_dat = pdat(3, :);

plot_dat = false;
clust_vis = true;

if clust_vis
    pdat_labels = LSC(pdat', 5, ['p', 400, 'numRep', 7]);
    ClusterVis(pdat', pdat_labels);
end

%% Original plot func
if plot_dat
    F(size(pdat, 2)) = struct('cdata',[],'colormap',[]); % movie
    writerObj = VideoWriter('examplemovie.avi');
    writerObj.Quality = 100;
    writerObj.FrameRate = 60;
    open(writerObj);
    pca_fig = figure('GraphicsSmoothing', 'on');
    xlabel('PCA 1', 'FontSize', 18); ylabel('PCA 2', 'FontSize', 18); zlabel('PCA 3', 'FontSize', 18);
    set(gca,'BoxStyle','full','Box','on', 'Position', [0.25, 0.25, 0.5, 0.5]);
    axis([min(p1_dat), max(p1_dat), min(p2_dat), max(p2_dat), min(p3_dat), max(p3_dat)])
    hold on;
    h = animatedline('Color', 'c', 'LineWidth', 0.1, 'LineStyle', '-', 'MaximumNumPoints', 100);
    tic;
    deg = 1;
    saved_color = 'b';
    color1 = 'b';
    for ii = 1:size(pdat, 2)
        ii
        time_data = t_linspace(ii + 1);
        a = find( diff(sign(CL(:,1)-time_data)) == 2);
        b = find( diff(sign(dec_sine(:,1)-time_data)) == 2);
        c = find( diff(sign(inc_sine(:,1)-time_data)) == 2);
        d = find( diff(sign(no_sim(:,1)-time_data)) == 2);
        e = find( diff(sign(tugs_ol(:,1)-time_data)) == 2);
        f = find( diff(sign(inf_sine(:,1)-time_data)) == 2);
        saved_color = color1;
        if (time_data > CL(a,1)) & (time_data < CL(a,1) + CL(a,2))
            color1 = 'r';
        elseif (time_data > dec_sine(b,1)) & (time_data < dec_sine(b,1) + dec_sine(b,2))
            color1 = 'm';
        elseif (time_data > inc_sine(c,1)) & (time_data < inc_sine(c,1) + inc_sine(c,2))
            color1 = 'k';
        elseif (time_data > inf_sine(f,1)) & (time_data < inf_sine(f,1) + inf_sine(f,2))
            color1 = 'c';
        elseif (time_data > no_sim(d,1)) & (time_data < no_sim(d,1) + no_sim(d,2))
            color1 = 'b';
        elseif (time_data > tugs_ol(e,1)) & (time_data < tugs_ol(e,1) + tugs_ol(e,2))
            color1 = 'y';
        else
            color1 = 'g';
        end

        p1 = pdat(1, ii);
        p2 = pdat(2, ii);
        p3 = pdat(3, ii);
        pca_plot(ii) = plot3(p1, p2, p3, 'Marker', '*', 'Color', color1, 'MarkerSize', 3, 'LineStyle', '-');

        if saved_color ~= color1
            set(pca_plot(1:ii-1), 'MarkerSize', 0.5);
        end

        deg = mod(deg + 0.25, 360);
        addpoints(h, pdat(1, ii), pdat(2, ii), pdat(3, ii));

        b = toc;
        if b > (1.0/1000)
            view([deg + 5, 15 + sin(deg/60)])
            drawnow update;
            F(:,ii) = getframe(gcf);
            writeVideo(writerObj,F(:,ii))
            tic;
        end
    end
end

