function n_frames = generate_feature(s, label, flag)
    % s: specification
    % train_info: labeling info of the training dataset
    % test_info: labeling info of the testing dataset
    
    if strcmp(flag, 'training')
        n = s.training_examples;
    elseif strcmp(flag, 'testing')
        n = s.testing_examples;
    else
        error('flag = <training/testing>');
    end
    assert(length(label) == n);
    

    path_spectrum = fullfile(s.root, flag, 'spectrum');
    if ~exist(path_spectrum, 'dir')
        mkdir(path_spectrum);
    end
    delete(fullfile(path_spectrum, '*.mat'));

    path_ideal = fullfile(s.root, flag, 'ideal_reconstruct');
    if ~exist(path_ideal, 'dir')
        mkdir(path_ideal);
    end
    delete(fullfile(path_ideal, '*.wav'));
    
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
        % temp = label(i).path;
        % audiowrite([temp(1:end-4) '-decomp.wav'], [y speech_component noise_component], s.sample_rate, 'BitsPerSample', 32);
        
        % calculate ideal-ratio masks
        nfft = s.feature.frame_length;
        hop = s.feature.hop_length;
        win = sqrt(hann(nfft,'periodic'));
        
        h_mix = stft2(y.', nfft, hop, 0, win);
        h_speech = stft2(speech_component.', nfft, hop, 0, win);
        h_noise = stft2(noise_component.', nfft, hop, 0, win);
        
        bm = abs(h_speech)./(abs(h_speech)+abs(h_noise));
        spec = abs(h_mix);
        save(fullfile(path_spectrum,['s_' num2str(i) '.mat']), 'bm', 'spec');
        
        
        % reconstruct based on ideal-ratio mask for top performance
        ideal_reconstruct = bm .* h_mix;
        % ideal_noise = (1-bm) .* h_mix; % don't care
        scale = 2;
        speech_best_recovered = stft2(ideal_reconstruct, nfft, hop, 0, win) * scale;
        audiowrite(fullfile(path_ideal,['s_' num2str(i) '.wav']), speech_best_recovered, s.sample_rate, 'BitsPerSample', 32);
        n_frames = n_frames + size(bm,2);
    end
end