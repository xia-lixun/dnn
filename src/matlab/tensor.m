function tensor(s, flag)

    path_spectrum = fullfile(s.root, flag, 'spectrum');
    dataset = dir(fullfile(path_spectrum, '*.mat'));
    
    % prepare tensor folder
    path_tensor = fullfile(s.root, flag, 'tensor');
    if ~exist(path_tensor, 'dir')
        mkdir(path_tensor);
    end
    delete(fullfile(path_tensor, '*.mat'));

    % convert spectrum to tensor
    for i = 1:length(dataset)
        load(fullfile(path_spectrum, dataset(i).name), 'ratiomask_mel', 'magnitude_mel');
        variable = single(sliding(magnitude_mel, (s.feature.context_span-1)/2, s.feature.nat_frames));
        label = single(ratiomask_mel);
        
        temp = split(dataset(i).name, '+');
        save(fullfile(path_tensor, ['t_' temp{1} '.mat']), 'label', 'variable', '-v6');
        clear ratiomask_mel
        clear magnitude_mel
    end
    
end
