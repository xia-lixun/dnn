function y = sdr_benchmark(path_evaluate, path_golden)

    sdr_db = 0;
    wav_eval = dir(fullfile(path_evaluate, '*.wav'));

    for i = 1:length(wav_eval)
        [dat_eval, fs] = audioread(fullfile(path_evaluate,wav_eval(i).name));
        [dat_gold, fs] = audioread(fullfile(path_golden,wav_eval(i).name));
        sdr_db = sdr_db + signal_to_distortion_ratio(dat_eval(:,1), dat_gold(:,1));
    end
    y = sdr_db / length(wav_eval);
end
