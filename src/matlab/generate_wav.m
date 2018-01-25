function [mix_info, mix_log] = generate_wav(s, layout, flag)

if strcmp(flag, 'training')
    mix_seconds = s.training_seconds;
elseif strcmp(flag, 'testing')
    mix_seconds = s.testing_seconds;
else
    error('flag = <training/testing>');
end
mix_sps = round(mix_seconds * s.sample_rate);
mix_count = 1;


% prepare the folder structure
if ~exist(s.root, 'dir')
    mkdir(s.root);
end
path_tt = fullfile(s.root, flag);
if ~exist(path_tt, 'dir')
    mkdir(path_tt);
end
path_tt_wav = fullfile(path_tt, 'wav');
if ~exist(path_tt_wav, 'dir')
    mkdir(path_tt_wav);
end
delete(fullfile(path_tt_wav, '*.wav'));


% split the layout as training/testing
% update: we must split in total time instead of by the ratio of instances.

n_speech = length(layout.speech.path);
%n_speech_train = round(n_speech * s.split_ratio_for_training);
[split_ratio_for_training, n_speech_train] = timesplit(layout.speech.length, s.split_ratio_for_training);
mix_log.speech.split_ratio_for_training = split_ratio_for_training;
mix_log.speech.n_speech_train = n_speech_train;


% iterate through each noise group
for i = 1:length(layout.noise)
    
    n_noise = length(layout.noise(i).path);
    %n_noise_train = round(n_noise * s.split_ratio_for_training);
    [split_ratio_for_training, n_noise_train] = timesplit(layout.noise(i).length, s.split_ratio_for_training);
    mix_log.noise(i).split_ratio_for_training = split_ratio_for_training;
    mix_log.noise(i).n_noise_train = n_noise_train;
    
    mix_sps_per_group = round(s.noisegroup(i).percent * 0.01 * mix_sps);
    mix_log.noise(i).mix_sps_setting = mix_sps_per_group;
    j = 0;
    while j < mix_sps_per_group
        
        spl_db = randselect(s.speech_level_db);
        snr = randselect(s.snr);
        if strcmp(flag, 'training')
            speech_rn = randi([1 n_speech_train]);
            noise_rn = randi([1 n_noise_train]);
        elseif strcmp(flag, 'testing')
            speech_rn = randi([n_speech_train+1 n_speech]);
            noise_rn = randi([n_noise_train+1 n_noise]);
        end
        
        speech_path = fullfile(s.speech, layout.speech.path{speech_rn});
        speech_level_max = layout.speech.level_max(speech_rn);
        speech_level_dbrms = layout.speech.level_dbrms(speech_rn);
        speech_length = layout.speech.length(speech_rn);
        
        noise_path = layout.noise(i).path{noise_rn};
        noise_level_max = layout.noise(i).level_max(noise_rn);
        noise_level_rms = layout.noise(i).level_rms(noise_rn);
        noise_level_med = layout.noise(i).level_med(noise_rn);
        noise_length = layout.noise(i).length(noise_rn);
        
        gains = zeros(2,1);
        
        % level the speech chosen to target
        [x,fs] = audioread(speech_path);
        assert(fs == s.sample_rate)
        assert(size(x,2) == 1)
        assert(size(x,1) == speech_length)
        
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
        assert(size(u,1) == noise_length)
        
        t = 10^((spl_db - snr)/20);
        if strcmp(s.noisegroup(i).type, 'impulsive')
            g = t / noise_level_max;
        elseif strcmp(s.noisegroup(i).type, 'stationary')
            g = t / noise_level_rms;
        elseif strcmp(s.noisegroup(i).type, 'nonstationary')
            g = t / noise_level_med;
        else
            error('wrong noise type detected');
        end
        if g * noise_level_max > 0.999
            g = 0.999 / noise_level_max;
        end
        u = u * g;
        gains(2) = g;
        
        
        % speech-noise time ratio control
        noise_id = path2id(noise_path, s.noise);
        speech_id = path2id(speech_path, s.speech);
        path_out = fullfile(path_tt_wav, [num2str(mix_count) '+' noise_id '+' speech_id '+' num2str(spl_db) '+' num2str(snr) '.wav']);
        
        mix_info(mix_count).path = path_out;
        mix_info(mix_count).gain = gains;
        mix_info(mix_count).src.speech = speech_path;
        mix_info(mix_count).src.noise = noise_path;
        
        eta = speech_length / noise_length;
       
        if eta > s.speech_noise_time_ratio
            % case speech length is too long, cyclic extend the noise
            noise_length_extended = round(speech_length / s.speech_noise_time_ratio);
            u_extended = cyclic_extend(u, noise_length_extended);
            insert_0 = randi([1 (noise_length_extended-speech_length+1)]);
            insert_1 = insert_0 + speech_length - 1;
            u_extended(insert_0:insert_1) = u_extended(insert_0:insert_1) + x;
            audiowrite(path_out, u_extended, s.sample_rate, 'BitsPerSample', 32);
            mix_info(mix_count).label = [insert_0 insert_1];
            j = j + length(u_extended);
            
        elseif eta < s.speech_noise_time_ratio
            % case when speech is too short for the noise, extend the
            % noise, we don't do cyclic extention with speech, but rather
            % scattering multiple copies in the noise clip.
            speech_length_total = round(noise_length * s.speech_noise_time_ratio);
            lambda = speech_length_total / speech_length;
            lambda_1 = floor(lambda) - 1.0;
            lambda_2 = lambda - floor(lambda) + 1.0;
            
            speech_length_extended = round(speech_length * lambda_2);
            x_extended = cyclic_extend(x, speech_length_extended);
            % Obs! speech extended cound have the same length of original
            
            partition_size = round(noise_length / lambda);
            partition = zeros(lambda_1+1, 1);
            for k = 1:lambda_1
                partition(k) = partition_size;
            end
            partition(end) = noise_length - (lambda_1 * partition_size);
            assert(partition(end) >= partition_size)
            partition = partition(randperm(length(partition)));
            [b_0, b_1] = borders(partition);
            
            labels = zeros(1, (lambda_1+1)*2);
            for k = 1:lambda_1+1
                if partition(k) > partition_size
                    insert_0 = randselect(b_0(k):b_1(k)-speech_length_extended+1);
                    insert_1 = insert_0 + speech_length_extended - 1;
                    u(insert_0:insert_1) = u(insert_0:insert_1) + x_extended;
                    labels((k-1)*2+1) = insert_0;
                    labels((k-1)*2+2) = insert_1;
                else
                    insert_0 = randselect(b_0(k):b_1(k)-speech_length+1);
                    insert_1 = insert_0 + speech_length - 1;
                    u(insert_0:insert_1) = u(insert_0:insert_1) + x;
                    labels((k-1)*2+1) = insert_0;
                    labels((k-1)*2+2) = insert_1;
                end
            end
            audiowrite(path_out, u, s.sample_rate, 'BitsPerSample', 32);
            mix_info(mix_count).label = labels;
            j = j + length(u);
            
        else
            % if eta hit the value precisely...
            insert_0 = randselect(1:noise_length-speech_length+1);
            insert_1 = insert_0 + speech_length - 1;
            u(insert_0:insert_1) = u(insert_0:insert_1) + x;
            audiowrite(path_out, u, s.sample_rate, 'BitsPerSample', 32);
            mix_info(mix_count).label = [insert_0 insert_1];
            j = j + length(u);
        end
        mix_count = mix_count + 1;
    end
    mix_log.noise(i).mix_sps_get = j;
end

end







function [y,offset] = timesplit(x, ratio)
% x: positive integer array, for example [2 3 5 7 9]
% ratio: point that splits the total amount
% this function tries to find such best split point.
    assert(ratio > 0.0 && ratio < 1.0);
    xs = cumsum(x);
    [temp, offset] = min(abs(xs / xs(end) - ratio));
    y = xs(offset)/xs(end);
end




function y = randselect(x)
% select one element from array x, randomly.
    y = x(randi([1 length(x)]));
end


function y = path2id(path, root)
% extract file id from path information
%   path = 'D:\5-Workspace\GoogleAudioSet\Engine\m0034_+H3HGFDkd43.wav'
%   root = 'D:\5-Workspace\GoogleAudioSet\'
%   y = 'Engine+m0034_+H3HGFDkd43'
m = length(root);
n = length('.wav');
y = replace(path(m+1:end-n),'\', '+');
end


function [start, stop] = borders(partition)
% generate borders based on length spefication
%   x = [3; 9; 11; 7]
%   borders = 1 .. 3
%             4 .. 12
%            13 .. 23
%            24 .. 30
    stop = cumsum(partition);
    start = [1; 1+stop(1:end-1)];
end