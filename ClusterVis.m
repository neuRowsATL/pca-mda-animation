function ClusterVis( projected_data, class_label_vector, outfile, thresh )
%ClusterVis : Creates a 3d visualization of clustered data
%   input: data separated into columns; each is a separate class
%   output: saves a movie as './examplemovie.avi'

% transpose projected_data
if length(projected_data) > size(projected_data, 1)
    projected_data = projected_data';
end

F(size(projected_data, 2)) = struct('cdata',[],'colormap',[]); % movie
if exist('outfile', 'var')
    writerObj = VideoWriter(outfile);
elseif ~exist('outfile', 'var')
    writerObj = VideoWriter('example_movie.avi');
end
writerObj.Quality = 100;
writerObj.FrameRate = 60;
open(writerObj);

p1_dat = projected_data(:, 1);
p2_dat = projected_data(:, 2);
p3_dat = projected_data(:, 3);

figure('GraphicsSmoothing', 'on', 'Renderer', 'opengl');
xlabel('PCA 1', 'FontSize', 18); ylabel('PCA 2', 'FontSize', 18); zlabel('PCA 3', 'FontSize', 18);
set(gca,'BoxStyle','full','Box','on', 'Position', [0.25, 0.25, 0.5, 0.5]);
set(gcf, 'OuterPosition', [400, 400, 900, 600])
axis([min(p1_dat)-0.5, max(p1_dat), min(p2_dat)-0.5, max(p2_dat), min(p3_dat)-1.0, max(p3_dat)])
hold on;

h = animatedline('Color', 'b', 'LineWidth', 0.5, 'LineStyle', '-', 'MaximumNumPoints', 10);

tic;
deg = 1;

saved_color = 'b';
colors = ['r', 'b', 'k', 'm', 'c', 'g', 'w'];
class_means = zeros(length(colors), 6);
UpdateRate = 1e-7; % smaller is slower
chunkData = round(length(projected_data)/5);
saved_val = 1;
oldvals = [];
plotted = zeros(length(colors));

thresh = thresh; % threshold for number of standard deviations tolerated

for ii = 1:size(projected_data, 1)
    ii
    cidx=class_label_vector(ii);
    color1 = colors(cidx);
    
    p1 = (projected_data(ii, 1));
    p2 = (projected_data(ii, 2));
    p3 = (projected_data(ii, 3));

    class_means(cidx, 4) = class_means(cidx, 4) + 1; % number of iterations
    class_means(cidx, 1) = (class_means(cidx, 1) + p1 ) / class_means(cidx,4); % p1 mean
    class_means(cidx, 2) = (class_means(cidx, 2) + p2 ) / class_means(cidx,4); % p2 mean
    class_means(cidx, 3) = (class_means(cidx, 3) + p3 ) / class_means(cidx,4); % p3 mean
    
    for cx=1:3
        if ii > 1
            if class_means(cidx, cx) >= thresh*std(oldvals(:, cidx, cx)) + mean(oldvals(:, cidx, cx)) || class_means(cidx, cx) <= mean(oldvals(:, cidx, cx)) - thresh*std(oldvals(:, cidx, cx))
                pca_plot(ii) = plot3(class_means(cidx, 1), class_means(cidx, 2), class_means(cidx, 3), 'Marker', 'o', 'MarkerFaceColor', color1, 'MarkerEdgeColor', 'b', 'MarkerSize', 10);
                addpoints(h, class_means(cidx, 1), class_means(cidx, 2), class_means(cidx, 3));
                plotted(cidx) = 1;
            end
            if plotted(cidx) == 0 % check: has this class mean already been plotted?
                pca_plot(ii) = plot3(class_means(cidx, 1), class_means(cidx, 2), class_means(cidx, 3), 'Marker', 'o', 'MarkerFaceColor', color1, 'MarkerEdgeColor', 'b', 'MarkerSize', 10);
                addpoints(h, class_means(cidx, 1), class_means(cidx, 2), class_means(cidx, 3));
                plotted(cidx) = 1;
            end
        end
    end
    
    deg = mod(deg + 0.25, 360);

    b = toc;
    if b > UpdateRate
        view([deg + 5, 15 + sin(deg/60)])
        drawnow update;
        F(:,ii) = getframe(gcf);
        writeVideo(writerObj,F(:,ii))
        tic;
    end
    
    oldvals(ii, :, :) = class_means(:, 1:3); % save all means
    
end

end

