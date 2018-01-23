function mix_info = generate_wav(s, layout, flag)

if flag == 'training'
    n_mix = s.training_examples;
elseif flag == 'testing'
    n_mix = s.testing_examples;
else
    error('flag = <training/testing>');
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


% iterate through each noise group

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
        
        
        % speech-noise time control
        speech_id = replace(speech_path(length(s.speech)+1:end), '\', '+');
        noise_id = replace(noise_path(length(s.noise)+1:end), '\', '+');
        path_out = [s.root '\' num2str(n_mix_count) '+' noise_id(1:end-4) '+' speech_id(1:end-4) '+' num2str(spl_db) '+' num2str(snr) '.wav'];
        
        gain(path_out) = gains;
        source(path_out) = [{speech_path} {noise_path}];
        eta = speech_length / noise_length;
        
        % speech-noise time ratio control
        noise_id = path2id(noise_path, s.noise);
        speech_id = path2id(speech_path, s.speech);
        path_out = fullfile(path_tt_wav, [num2str(n_mix_count) '+' noise_id '+' speech_id '+' num2str(spl_db) '+' num2str(snr) '.wav']);
        
        mix_info(n_mix_count).path = path_out;
        mix_info(n_mix_count).gain = gains;
        mix_info(n_mix_count).src.speech = speech_path;
        mix_info(n_mix_count).src.noise = noise_path;
        
        eta = speech_length / noise_length;
       
        if eta > s.speech_noise_time_ratio
            % case speech length is too long, cyclic extend the noise
            noise_length_extended = round(speech_length / s.speech_noise_time_ratio);
            u_extended = cyclic_extend(u, noise_length_extended);
            insert_0 = randi([1 (noise_length_extended-speech_length+1)]);
            insert_1 = insert_0 + speech_length - 1;
            u_extended(insert_0:insert_1) = u_extended(insert_0:insert_1) + x;
            audiowrite(path_out, u_extended, s.sample_rate, 'BitsPerSample', 32);
            mix_info(n_mix_count).label = [insert_0 insert_1];
            
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
            mix_info(n_mix_count).label = labels;
            
        else
            % if eta hit the value precisely...
            insert_0 = randselect(1:noise_length-speech_length+1);
            insert_1 = insert_0 + speech_length - 1;
            u(insert_0:insert_1) = u(insert_0:insert_1) + x;
            audiowrite(path_out, u, s.sample_rate, 'BitsPerSample', 32);
            mix_info(n_mix_count).label = [insert_0 insert_1];
        end
        n_mix_count = n_mix_count + 1;
    end
end

fid = fopen(fullfile(s.root, flag, 'mix_info.json'),'w');
fprintf(fid, jsonencode(mix_info));
fclose(fid);

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


function y = cyclic_extend(x, n)
% x y belongs to 1-d array
    y = zeros(n,1);
    m = length(x);
    for i = 1:n
        y(i) = x(mod((i-1), m) + 1);
    end
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