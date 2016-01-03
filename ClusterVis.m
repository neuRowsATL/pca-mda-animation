function ClusterVis( projected_data, class_label_vector )
%ClusterVis : Creates a 3d visualization of clustered data
%   input: data separated into columns; each is a separate class
%   output: saves a movie as './examplemovie.avi'

F(size(projected_data, 2)) = struct('cdata',[],'colormap',[]); % movie
writerObj = VideoWriter('examplemovie.avi');
writerObj.Quality = 100;
writerObj.FrameRate = 30;
open(writerObj);

p1_dat = projected_data(:, 1);
p2_dat = projected_data(:, 2);
p3_dat = projected_data(:, 3);

figure('GraphicsSmoothing', 'on', 'Renderer', 'opengl');
xlabel('PCA 1', 'FontSize', 18); ylabel('PCA 2', 'FontSize', 18); zlabel('PCA 3', 'FontSize', 18);
set(gca,'BoxStyle','full','Box','on', 'Position', [0.25, 0.25, 0.5, 0.5]);
set(gcf, 'OuterPosition', [400, 400, 900, 600])
axis([min(p1_dat), max(p1_dat), min(p2_dat), max(p2_dat), min(p3_dat), max(p3_dat)])
hold on;

h = animatedline('Color', 'g', 'LineWidth', 1, 'LineStyle', '-', 'MaximumNumPoints', 75);

tic;
deg = 1;

saved_color = 'b';
colors = ['r', 'b', 'k', 'm', 'c', 'g'];
UpdateRate = 1e-8;
chunkData = round(length(projected_data)/5);
saved_val = 1;

for ii = 1:size(projected_data, 1)
    ii
    cidx=class_label_vector(ii);
    color1 = colors(cidx);
    
    p1 = projected_data(ii, 1);
    p2 = projected_data(ii, 2);
    p3 = projected_data(ii, 3);
    
    pca_plot(ii) = plot3(p1, p2, p3, 'Marker', '.', 'Color', color1, 'MarkerSize', 20);
    
    if ii >= saved_val + chunkData
        saved_color = color1;
        saved_val = ii;
        set(pca_plot(1:ii-1), 'MarkerSize', 5);
    end
    
    deg = mod(deg + 0.25, 360);
    addpoints(h, projected_data(ii, 1), projected_data(ii, 2), projected_data(ii, 3));

    b = toc;
    if b > UpdateRate
        view([deg + 5, 15 + sin(deg/60)])
        drawnow update;
        F(:,ii) = getframe(gcf);
        writeVideo(writerObj,F(:,ii))
        tic;
    end
    
end

end

