% data mixing scripts for selective noise reduction
% lixun.xia2@harman.com
% 2018-01-16
function y = mix()
    s = specification();
    data = index(s);
    y = generate_wav(s, data, 'training');
end



function m = index(s)
% build speech/noise data layout for mixing.

rng(s.random_seed);
tmp = rand(100);
clear tmp;

% populate noise group folders and calculate levels
for i = 1:length(s.noisegroup)
    
    path_group = fullfile(s.noise, s.noisegroup(i).name);
    group =  dir([path_group '/**/*.wav']);
    % populate noise examples and shuffle the sequence within each group
    for j = 1:length(group)    
        layout.noise(i).path(j) = cellstr(fullfile(group(j).folder, group(j).name));  
    end
    layout.noise(i).path = layout.noise(i).path(randperm(length(group)));
    
    % calculate the levels of the shuffled examples
    for j = 1:length(group)
        
        [x, fs] = audioread(layout.noise(i).path{j});
        assert(fs == s.sample_rate)
        assert(size(x,2) == 1)
        y = abs(x);
        layout.noise(i).level_max(j) = max(y);
        layout.noise(i).level_rms(j) = rms(x);
        layout.noise(i).level_med(j) = median(y);
        layout.noise(i).length(j) = size(y,1);
    end
end


% load clean speech level info
fid = fopen(fullfile(s.speech, 'index.level'));
c = textscan(fid, '%s %f %f %f', 'Delimiter',',', 'CommentStyle','#');
fclose(fid);

shuffle = randperm(length(c{1}));
layout.speech.path = c{1};
layout.speech.level_max = c{2};
layout.speech.level_dbrms = c{3};
layout.speech.length = c{4};

layout.speech.path = layout.speech.path(shuffle);
layout.speech.level_max = layout.speech.level_max(shuffle);
layout.speech.level_dbrms = layout.speech.level_dbrms(shuffle);
layout.speech.length = layout.speech.length(shuffle);



m = layout;
end

