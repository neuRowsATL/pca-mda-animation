
% display stuff
color_and_symbol_5_classes = ['k.'; 'cd'; 'go'; 'gx'; 'b^'; 'bs'; 'm*';];
color_and_symbol_5_classes = ['k.'; 'go'; 'b^'; 'm*'; 'cd'];
color_and_symbol = color_and_symbol_5_classes;

clear h;

nr_classes = 5;

figure
for jj = 1:5
    subplot(3, 2, jj);
    hold on;
    for ii = 1:nr_classes
        p1 = find(g_training_set == ii);
        h(ii) = plot(training_set(p1), y_train(p1, jj), color_and_symbol_5_classes(ii, :));
        p1 = find(g_test_set == ii);
        plot(test_set(p1), y_test(p1, jj), ['r' color_and_symbol_5_classes(ii, 2)]);
    end
    hold off;
end


if(size(Disc, 2) > 2)

    counter = 0;
    clear h2
    if(nr_features >= 3)


        figure; hold on;

        clear vs v;

        for ii = 1:nr_classes
            counter = counter + 1;

            p1 = find(g_training_set == ii);
            h2(counter) = plot3(y_train(p1, select_projection(1)), y_train(p1, select_projection(2)), y_train(p1, select_projection(3)), ['k' color_and_symbol(ii, 2)], 'MarkerSize', 10);
        end

        for ii = 1:nr_classes 
            counter = counter + 1;

            p1 = find(g_test_set == ii);

            h2(counter) = plot3(y_test(p1, select_projection(1)), y_test(p1, select_projection(2)), y_test(p1, select_projection(3)), ['r' color_and_symbol(ii, 2)], 'MarkerSize', 10);
        end
    end


    for ii = 1:nr_classes

        p1 = find(g_training_set == ii);
        centered_projection1 = y_train(p1, select_projection(1:3));

        [u, s, v] = svd(centered_projection1);
        v = v(1:3, 1:3);
        vs(:, :, ii) = v;

        centered_projection2 = centered_projection1*v;

        factor2 = 2;

        max1 = max([std(centered_projection2(:, 1)) std(centered_projection2(:, 2)) std(centered_projection2(:, 3))]);


        factor2_1 = factor2*std(centered_projection2(:, 1));
        factor2_2 = factor2*std(centered_projection2(:, 2));
        factor2_3 = factor2*std(centered_projection2(:, 3));

        p1 = min([factor2_1 factor2_2 factor2_3]); p1 = p1*0;
        factor2_1 = factor2_1 + p1;
        factor2_2 = factor2_2 + p1;
        factor2_3 = factor2_3 + p1;

        [x y z] = ellipsoid(mean(centered_projection2(:, 1)), mean(centered_projection2(:, 2)), mean(centered_projection2(:, 3)), ...
            factor2_1, factor2_2, factor2_3, 1000);


        p1 = [x(1:end)' y(1:end)' z(1:end)'];
        p2 = p1*inv(v);
        x1 = zeros(size(x)); y1 = zeros(size(y)); z1 = zeros(size(z));
        x1(1:end) = p2(:, 1);
        y1(1:end) = p2(:, 2);
        z1(1:end) = p2(:, 3);
        h = surfl(x1, y1, z1);
        set(h, 'FaceColor', color_and_symbol(ii, 1), 'LineStyle', 'none');

        counter = counter + 1;
        h2(counter) = h;
    end


    
    camlight headlight;
    alpha(0.3);


    hold off;

end

