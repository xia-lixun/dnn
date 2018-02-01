% data mixing scripts for selective noise reduction
% lixun.xia2@harman.com
% 2018-01-16
function s = specification()

specification.root = 'D:\7-Workspace\';
specification.noise = 'D:\5-Workspace\GoogleAudioSet\NoSpeech\';
specification.speech = 'D:\5-Workspace\Voice\';

specification.speech_noise_time_ratio = 0.6;
specification.split_ratio_for_training = 0.75;   % this is also time ratio
specification.speech_level_db = [-22.0 -32.0 -42.0];
specification.snr = [20.0 15.0 10.0 5.0 0.0 -5.0];

specification.training_seconds = 5000;
specification.testing_seconds = 2000;
specification.sample_rate = 16000;
specification.random_seed = 1234;

specification.feature.frame_length = 512;
specification.feature.hop_length = 128;
specification.feature.context_span = 23;
specification.feature.nat_frames = 14;
specification.feature.mel_filter_banks = 26;


%------------------------noise database----------------------------
specification.noisegroup(1).name = 'Accelerating, revving, vroom';
specification.noisegroup(1).percent = 10.0;
specification.noisegroup(1).type = 'stationary';

specification.noisegroup(2).name = 'Air brake';
specification.noisegroup(2).percent = 10.0;
specification.noisegroup(2).type = 'stationary';

specification.noisegroup(3).name = 'Air conditioning';
specification.noisegroup(3).percent = 10.0;
specification.noisegroup(3).type = 'stationary';

specification.noisegroup(4).name = 'Air horn, truck horn';
specification.noisegroup(4).percent = 10.0;
specification.noisegroup(4).type = 'stationary';

specification.noisegroup(5).name = 'Aircraft';
specification.noisegroup(5).percent = 10.0;
specification.noisegroup(5).type = 'stationary';

specification.noisegroup(6).name = 'Aircraft engine';
specification.noisegroup(6).percent = 10.0;
specification.noisegroup(6).type = 'stationary';

specification.noisegroup(7).name = 'Alarm';
specification.noisegroup(7).percent = 10.0;
specification.noisegroup(7).type = 'stationary';

specification.noisegroup(8).name = 'Alarm clock';
specification.noisegroup(8).percent = 10.0;
specification.noisegroup(8).type = 'stationary';

specification.noisegroup(9).name = 'Ambulance (siren)';
specification.noisegroup(9).percent = 10.0;
specification.noisegroup(9).type = 'stationary';

specification.noisegroup(10).name = 'Applause';
specification.noisegroup(10).percent = 10.0;
specification.noisegroup(10).type = 'stationary';

s = specification;
end