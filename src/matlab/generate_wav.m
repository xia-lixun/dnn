function m = generate_wav(s, layout, flag)

if flag == 'training'
    n_mix = s.training_examples;
elseif flag == 'testing'
    n_mix = s.testing_examples;
else
    error('flag = <training/testing>');
    % todo: add validation set
end
n_mix_count = 1;


% prepare the folder structure
mkdir(s.root);
path_tt = fullfile(s.root, flag);
mkdir(path_tt);
path_tt_wav = fullfile(path_tt, 'wav');
mkdir(path_tt_wav);
delete(fullfile(path_tt_wav, '*.wav'));


% split the layout as training/testing
n_speech = length(layout.speech.path);
n_speech_train = round(n_speech * s.split_ratio_for_training);


% going through each noise group
for i = 1:length(layout.noise)
    
    n_noise = length(layout.noise(i).path);
    n_noise_train = round(n_noise * s.split_ratio_for_training);
    n_mix_per_group = round(s.noisegroup(i).percent * 0.01 * n_mix);
    
    for j = 1:n_mix_per_group
        
        spl_db = randselect(s.speech_level_db);
        snr = randselect(s.snr);
        if flag == 'training' 
            speech_rn = randi([1 n_speech_train]);
            noise_rn = randi([1 n_noise_train]);
        elseif flag == 'testing'
            speech_rn = randi([n_speech_train+1 n_speech]);
            noise_rn = randi([n_noise_train+1 n_noise]);
        end
        
        speech_path = layout.speech.path{speech_rn};
        speech_level_max = layout.speech.level_max(speech_rn);
        speech_level_dbrms = layout.speech.level_dbrms(speech_rn);
        speech_length = layout.speech.length(speech_rn);
        
        noise_path = layout.noise(i).path{noise_rn};
        noise_level_max = layout.noise.level_max(noise_rn);
        noise_level_rms = layout.noise.level_rms(noise_rn);
        noise_level_med = layout.noise.level_med(noise_rn);
        noise_length = layout.noise.length(noise_rn);
        
        gains = zeros(2,1);
        
        % level the speech chosen to target
        [x,fs] = audioread(speech_path);
        assert(fs == s.sample_rate)
        assert(size(x,2) == 1)
        g = 10^((spl_db - speech_level_dbrms)/20);
        if g * speech_level_max > 0.999
            g = 0.999 / speech_level_max;
            spl_db = speech_level_dbrms + 20*log10(g + 1e-7);
        end
        x = x * g;
        gains(1) = g;
        
        % calculate noise level based on speech level and snr
        [u,fs] = audioread(noise_path);
        assert(fs == s.sample_rate)
        assert(size(u,2) == 1)
        
        t = 10^((spl_db - snr)/20);
        if s.noisegroup(i).type == 'impulsive'
            g = t / noise_level_max;
        elseif s.noisegroup(i).type == 'stationary'
            g = t / noise_level_rms;
        elseif s.noisegroup(i).type == 'nonstationary'
            g = t / noise_level_med;
        else
            error('wrong noise type detected');
        end
        if g * noise_level_max > 0.999
            g = 0.999 / noise_level_max;
        end
        u = u * g;
        gains(2) = g;
        
        
    end
end

end


% select one element from array x, randomly.
function y = randselect(x)
    y = x(randi([1 length(x)]));
end