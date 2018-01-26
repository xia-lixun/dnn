function tensor(s, label, mu, std, flag)

    n = length(label);
    path_spectrum = fullfile(s.root, flag, 'spectrum');
    
    % prepare tensor folder
    path_tensor = fullfile(s.root, flag, 'tensor');
    if ~exist(path_tensor, 'dir')
        mkdir(path_tensor);
    end
    delete(fullfile(path_tensor, '*.mat'));

    % convert spectrum to tensor
    for i = 1:n
        load(fullfile(path_spectrum,['s_' num2str(i) '.mat']), 'bm', 'spec');
        spec = sliding((spec - mu)./std, (s.feature.context_span-1)/2, s.feature.nat_frames);
        bm = bm.';
        spec = spec.';
        save(fullfile(path_tensor,['t_' num2str(i) '.mat']), 'bm', 'spec');
    end
    
end
