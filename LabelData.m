function pdat_labels = LabelData( pdat, type )
%LabelData categorizes projected data (# of classes determined by columns)
%   input: data in vector
%   output: returns 1 x length(data) array with labels

if type == 1
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
    
elseif type == 2
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

cbco_data = cbco_data';
    firing_times = cbco_data;

    nvar = max(firing_times(2, :));

    nr_points = 10e2 - 1;
    t_linspace = linspace(min(firing_times(1, :)), max(firing_times(1, :)), nr_points);
    delta_t = t_linspace(2) - t_linspace(1);
    t_linspace = [t_linspace(1) - delta_t, t_linspace, t_linspace(end) + delta_t];
    nr_points = nr_points + 1;
    pdat_labels = zeros(1, size(pdat, 2));
    color1 = 'b';
    for ii=1:length(pdat)
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
        pdat_labels(ii) = color1;
    end
end


end

