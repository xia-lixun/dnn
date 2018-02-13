function [mu_bm, mu_spec, std_bm, std_spec] = statistics(s, n_frames, flag)

    path_spectrum = fullfile(s.root, flag, 'spectrum');
    dataset = dir(fullfile(path_spectrum, '*.mat'));
    
    % find out dimensions of ratiomask and spectrum
    load(fullfile(path_spectrum, dataset(1).name), 'ratiomask_mel', 'magnitude_mel');
    mu_bm = zeros(size(ratiomask_mel,1),1);
    std_bm = zeros(size(ratiomask_mel,1),1);
    mu_spec = zeros(size(magnitude_mel,1),1);
    std_spec = zeros(size(magnitude_mel,1),1);
    clear ratiomask_mel;
    clear magnitude_mel;
    
    for i = 1:length(dataset)
        load(fullfile(path_spectrum, dataset(i).name), 'ratiomask_mel', 'magnitude_mel');
        mu_bm = mu_bm + sum(ratiomask_mel,2);
        mu_spec = mu_spec + sum(magnitude_mel,2);
    end
    mu_bm = mu_bm / n_frames;
    mu_spec = mu_spec / n_frames;
    
    
    % find out std of bm and spec    
    for i = 1:length(dataset)
        load(fullfile(path_spectrum, dataset(i).name), 'ratiomask_mel', 'magnitude_mel');
        std_bm = std_bm + sum((ratiomask_mel - mu_bm).^2,2);
        std_spec = std_spec + sum((magnitude_mel - mu_spec).^2,2);
    end
    std_bm = sqrt(std_bm / (n_frames-1));
    std_spec = sqrt(std_spec / (n_frames-1));
end