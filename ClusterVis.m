function [ ] = Untitled2( projected_data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

F(size(projected_data, 2)) = struct('cdata',[],'colormap',[]); % movie
writerObj = VideoWriter('examplemovie.avi');
open(writerObj);

p1_dat = projected_data(1, :);
p2_dat = projected_data(2, :);
p3_dat = projected_data(3, :);


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
for ii = 1:size(projected_data, 2)
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
        
    p1 = projected_data(1, ii);
    p2 = projected_data(2, ii);
    p3 = projected_data(3, ii);
    pca_plot(ii) = plot3(p1, p2, p3, 'Marker', '*', 'Color', color1, 'MarkerSize', 3, 'LineStyle', '-');
    
    if saved_color ~= color1
        set(pca_plot(1:ii-1), 'MarkerSize', 0.5);
    end
    
    deg = mod(deg + 0.25, 360);
    addpoints(h, projected_data(1, ii), projected_data(2, ii), projected_data(3, ii));

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

