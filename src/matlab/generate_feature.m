function n_frames = generate_feature(s, label, flag)
    % s: specification
    % train_info: labeling info of the training dataset
    % test_info: labeling info of the testing dataset
    
    n = length(label);
    path_spectrum = fullfile(s.root, flag, 'spectrum');
    if ~exist(path_spectrum, 'dir')
        mkdir(path_spectrum);
    end
    delete(fullfile(path_spectrum, '*.mat'));

    path_decomposition = fullfile(s.root, flag, 'decomposition');
    path_dft_ideal = fullfile(s.root, flag, 'oracle', 'dft');
    path_mel_ideal = fullfile(s.root, flag, 'oracle', 'mel');
    if ~exist(path_decomposition, 'dir')
        mkdir(path_decomposition);
    end
    if ~exist(path_dft_ideal, 'dir')
        mkdir(path_dft_ideal);
    end
    if ~exist(path_mel_ideal, 'dir')
        mkdir(path_mel_ideal);
    end
    delete(fullfile(path_decomposition, '*.wav'));
    delete(fullfile(path_dft_ideal, '*.wav'));
    delete(fullfile(path_mel_ideal, '*.wav'));
    
    % convert spectrum to tensor, using mel filter bank
    mel = filter_banks(s.sample_rate, s.feature.frame_length, s.feature.mel_filter_banks, 0, s.sample_rate/2);
    mel_weight = sum(mel,2);
    n_frames = 0;
    
    for i = 1:n
        % retrieve the mix/clean speech/noise clip
        [y,fs] = audioread(label(i).path);
        [x,fs] = audioread(label(i).src.speech);
        %[u,fs] = audioread(label(i).src.noise);
        
        % restore speech/noise to target gains
        x = x * label(i).gain(1);
        %u = u * label(i).gain(2);
        
        % reconstruct the clean speech component
        speech_component = zeros(length(y),1);
        segment = length(label(i).label)/2;
        for j = 0:segment-1
            insert_0 = label(i).label(2*j+1);
            insert_1 = label(i).label(2*j+2);
            speech_component(insert_0:insert_1) = cyclic_extend(x,insert_1-insert_0+1);
        end
        
        % reconstruct the noise component
        noise_component = y - speech_component;
        noise_component = noise_component + (10^(-120/20)) * rand(size(noise_component));
        
        % write speech and noise components to
        % /root/<training|testing>/decomposition
        [t_dir, t_base, t_ext] = fileparts(label(i).path);
        audiowrite(fullfile(path_decomposition, [t_base t_ext]), [speech_component noise_component], s.sample_rate, 'BitsPerSample', 32);
        
        % calculate ideal-ratio masks
        nfft = s.feature.frame_length;
        hop = s.feature.hop_length;
        win = sqrt(hann(nfft,'periodic'));
        
        h_mix = stft2(y.', nfft, hop, 0, win);
        h_speech = stft2(speech_component.', nfft, hop, 0, win);
        h_noise = stft2(noise_component.', nfft, hop, 0, win);
        
        ratiomask_dft = abs(h_speech)./(abs(h_speech)+abs(h_noise));
        ratiomask_mel = (mel * ratiomask_dft) ./ mel_weight;
        
        magnitude_dft = abs(h_mix);
        magnitude_mel = log(mel * magnitude_dft + eps);
        
        % save spectrum for dnn training/validation
        save(fullfile(path_spectrum,[t_base '.mat']), 'ratiomask_mel', 'magnitude_mel');
        
        
        % reconstruct based on ideal-ratio mask for top performance
        % write best recovered speech in dft band to
        % /root/<training|testing>/oracle/dft
        % write best recovered speech in mel band to
        % /root/<training|testing>/oracle/mel
        
        oracle_dft = ratiomask_dft .* h_mix;
        oracle_mel = (mel.' * ratiomask_mel) .* h_mix;
        % oracle_noise = (1 - oracle_dft) .* h_mix; % don't care for now
        scale = 2;
        speech_recovered_dft = scale * stft2(oracle_dft, nfft, hop, 0, win);
        speech_recovered_mel = scale * stft2(oracle_mel, nfft, hop, 0, win);
        
        audiowrite(fullfile(path_dft_ideal,[t_base t_ext]), speech_recovered_dft, s.sample_rate, 'BitsPerSample', 32);
        audiowrite(fullfile(path_mel_ideal,[t_base t_ext]), speech_recovered_mel, s.sample_rate, 'BitsPerSample', 32);
        
        n_frames = n_frames + size(magnitude_mel,2);
    end
end