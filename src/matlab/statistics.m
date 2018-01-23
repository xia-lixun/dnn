function [mu_bm, mu_spec, std_bm, std_spec] = statistics(s, n_frames, flag)

    if strcmp(flag, 'training')
        n = s.training_examples;
    elseif strcmp(flag, 'testing')
        n = s.testing_examples;
    else
        error('flag = <training/testing>');
    end
    path_spectrum = fullfile(s.root, flag, 'spectrum');
    
    % find out mu of bm and spec
    mu_bm = zeros(s.feature.frame_length/2+1,1);
    mu_spec = zeros(s.feature.frame_length/2+1,1);
    
    for i = 1:n
        load(fullfile(path_spectrum,['s_' num2str(i) '.mat']), 'bm', 'spec');
        mu_bm = mu_bm + sum(bm,2);
        mu_spec = mu_spec + sum(spec,2);
    end
    mu_bm = mu_bm / n_frames;
    mu_spec = mu_spec / n_frames;
    
    
    % find out std of bm and spec
    std_bm = zeros(s.feature.frame_length/2+1,1);
    std_spec = zeros(s.feature.frame_length/2+1,1);
    
    for i = 1:n
        load(fullfile(path_spectrum,['s_' num2str(i) '.mat']), 'bm', 'spec');
        std_bm = std_bm + sum((bm - mu_bm).^2,2);
        std_spec = std_spec + sum((spec - mu_spec).^2,2);
    end
    std_bm = sqrt(std_bm / (n_frames-1));
    std_spec = sqrt(std_spec / (n_frames-1));
end