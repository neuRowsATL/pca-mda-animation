
% display stuff
% color_and_symbol_5_classes = ['y.'; 'go'; 'gx'; 'b^'; 'bs'; 'm*'; 'kd'];
color_and_symbol_5_classes = ['k.'; 'cd'; 'go'; 'gx'; 'b^'; 'bs'; 'm*';];

color_and_symbol_5_classes = ['k.'; 'go'; 'b^'; 'm*'; 'cd'];


% color_and_symbol = ['k.'; 'go'; 'gx'; 'g*'; 'bx'; 'b*'; 'bo'];
% color_and_symbol = ['k.'; 'gs'; 'cd'; 'go'; 'gx'; 'g*'; 'mp'; 'bs'; 'bo'; 'bx'; 'b*'];
color_and_symbol = color_and_symbol_5_classes;

clear h;

% g = [ones(size(experiment_id0)) (experiment_id0 + 1)];
% g_training_set = g(training_set);
% g_test_set = g(test_set);

nr_classes = 5;

% if(size(Disc, 2) > 2)
%     figure(1); hold on;
%     for ii = 1:nr_classes + 1
%         p1 = find(g_training_set == ii);
%         plot3(y_train(p1, select_projection(1)), y_train(p1, select_projection(2)), y_train(p1, select_projection(3)), color_and_symbol_5_classes(ii, :));
%
%         for jj = 1:length(p1)
%             text(y_train(p1(jj), select_projection(1))  + 0.1, y_train(p1(jj), select_projection(2))  + 0.1, y_train(p1(jj), select_projection(3)) + 0.1, num2str(p1(jj)));
%         end
%     end
%     hold off;
%
%     figure(1); hold on;
%     for ii = 1:nr_classes + 1
%         p1 = find(g_test_set == ii);
%         plot3(y_test(p1, select_projection(1)), y_test(p1, select_projection(2)) , y_test(p1, select_projection(3)), ['r' color_and_symbol_5_classes(ii, 2)]);
%     end
%
%     hold off;
% else
%     figure(1); hold on;
%     for ii = 1:nr_classes + 1
%         p1 = find(g_training_set == ii);
%         plot(y_train(p1, select_projection(1)), y_train(p1, select_projection(2)), color_and_symbol_5_classes(ii, :));
%         %
%         %         for jj = 1:length(p1)
%         %             text(y_train(p1(jj), select_projection(1))  + 0.1, y_train(p1(jj), select_projection(2))  + 0.1, num2str(p1(jj)));
%         %         end
%     end
%     %         hold off;
%
%     figure(1); hold on;
%     for ii = 1:nr_classes + 1
%         p1 = find(g_test_set == ii);
%         plot(y_test(p1, select_projection(1)), y_test(p1, select_projection(2)), ['r' color_and_symbol_5_classes(ii, 2)]);
%     end
%
%     hold off;
%
% end


figure
% for jj = 1:size(Disc, 2)
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
% subplot(3, 4, 12);
% legend(h, 'Rest', 'Air A 200ms', 'Metal Sound', 'Air 200ms', 'Air 400ms', 'Air 800ms', 'Shake', 'Drop 11 cm env A', 'Drop 5cm', 'Drop 11cm', 'Drop 31cm' );

if(size(Disc, 2) > 2)

    counter = 0;
    clear h2
    if(nr_features >= 3)


        figure; hold on;

        clear vs v;
        %         figure(4); hold on;
        for ii = 1:nr_classes
            counter = counter + 1;

            p1 = find(g_training_set == ii);
            %         h2(counter) = plot3(y_train(p1, select_projection(1)), y_train(p1, select_projection(2)), y_train(p1, select_projection(3)), color_and_symbol(ii, :));
            h2(counter) = plot3(y_train(p1, select_projection(1)), y_train(p1, select_projection(2)), y_train(p1, select_projection(3)), ['k' color_and_symbol(ii, 2)], 'MarkerSize', 10);
        end

        %         for ii = 1:size(y_train, 1)
        %             text(y_train(ii, 1), y_train(ii, 2), y_train(ii, 3), num2str(training_set(ii)), 'Color', 'k');
        %         end
        %         for ii = 1:size(y_test, 1)
        %             text(y_test(ii, 1), y_test(ii, 2), y_test(ii, 3), num2str(test_set(ii)), 'Color', 'r');
        %         end
        for ii = 1:nr_classes 
            counter = counter + 1;

            p1 = find(g_test_set == ii);
            %             if(ii == 1), p1 = p1(1:4);  else, p1 = p1(1); end

            h2(counter) = plot3(y_test(p1, select_projection(1)), y_test(p1, select_projection(2)), y_test(p1, select_projection(3)), ['r' color_and_symbol(ii, 2)], 'MarkerSize', 10);
        end
    end

    % y_train = -y_train;


    %%% Simulates re-mapping
    % % ii  = 2;
    % % p1 = find(g_training_set == ii);
    % % y_train_set(p1, 2) = y_train_set(p1, 2) - mean(y_train_set(p1, 2));
    % % y_train_set(p1, 3) = y_train_set(p1, 3) - mean(y_train_set(p1, 3));
    % %
    % % ii  = 3;
    % % p1 = find(g_training_set == ii);
    % % y_train_set(p1, 1) = y_train_set(p1, 1) - mean(y_train_set(p1, 1));
    % % y_train_set(p1, 3) = y_train_set(p1, 3) - mean(y_train_set(p1, 3));
    % %
    % % ii  = 4;
    % % p1 = find(g_training_set == ii);
    % % y_train_set(p1, 2) = y_train_set(p1, 2) - mean(y_train_set(p1, 2));
    % % y_train_set(p1, 1) = y_train_set(p1, 1) - mean(y_train_set(p1, 1));



    %     figure; hold on;
    for ii = 1:nr_classes
        %     for ii = 2:3

        p1 = find(g_training_set == ii);
        centered_projection1 = y_train(p1, select_projection(1:3));

        [u, s, v] = svd(centered_projection1);
        v = v(1:3, 1:3);
        vs(:, :, ii) = v;

        centered_projection2 = centered_projection1*v;

        factor2 = 2;

        max1 = max([std(centered_projection2(:, 1)) std(centered_projection2(:, 2)) std(centered_projection2(:, 3))]);

        %         [x y z] = ellipsoid(mean(centered_projection2(:, 1)), mean(centered_projection2(:, 2)), mean(centered_projection2(:, 3)), ...
        %             factor2*std(centered_projection2(:, 1)), factor2*std(centered_projection2(:, 2)), factor2*std(centered_projection2(:, 3)), 100);

        factor2_1 = factor2*std(centered_projection2(:, 1));
        factor2_2 = factor2*std(centered_projection2(:, 2));
        factor2_3 = factor2*std(centered_projection2(:, 3));

        p1 = min([factor2_1 factor2_2 factor2_3]); p1 = p1*0;
        factor2_1 = factor2_1 + p1;
        factor2_2 = factor2_2 + p1;
        factor2_3 = factor2_3 + p1;


        %     factor2_1 = factor2_1 + factor2_3/4;
        %     factor2_2 = factor2_2 + factor2_3/4;
        %     factor2_3 = factor2_3 + factor2_3/4;


        %         factor2_3 = mean([factor2_1 factor2_2 factor2_3]);
        %         factor2_2 = mean([factor2_1 factor2_2]);

        [x y z] = ellipsoid(mean(centered_projection2(:, 1)), mean(centered_projection2(:, 2)), mean(centered_projection2(:, 3)), ...
            factor2_1, factor2_2, factor2_3, 1000);

        %         [x y z] = ellipsoid(mean(centered_projection2(:, 1)), mean(centered_projection2(:, 2)), mean(centered_projection2(:, 3)), ...
        %             factor2*max1, factor2*max1, factor2*max1, 100);

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

    %     legend(h2, 'Rest - training points', 'Airpuff - training points', 'Drop  - training points', 'Shake - training points', ...
    %         'Rest - test points', 'Airpuff - test points', 'Drop  - test points', 'Shake - test points', ...
    %         'Rest ellipsoids', 'Airpuff  ellipsoids', 'Drop ellipsoids', 'Shake ellipsoids');

    % pause;

    theta1 = 30; phi1 = -10; view([theta1, phi1]);
    camlight headlight;
    alpha(0.3);


    hold off;

end

