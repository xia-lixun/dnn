% data mixing scripts for selective noise reduction
% lixun.xia2@harman.com
% 2018-01-16
function s = specification()

specification.root = 'D:\7-Workspace\';
specification.noise = 'D:\5-Workspace\GoogleAudioSet\NoSpeech\';
specification.speech = 'D:\5-Workspace\Voice\';

specification.speech_noise_time_ratio = 0.6;
specification.split_ratio_for_training = 0.7;
specification.speech_level_db = [-22.0 -32.0 -42.0];
specification.snr = [20.0 15.0 10.0 5.0 0.0 -5.0];
specification.training_examples = 20;
specification.testing_examples = 10;
specification.sample_rate = 16000;
specification.random_seed = 42;
specification.feature.frame_length = 512;
specification.feature.hop_length = 128;
specification.feature.context_span = 23;
specification.feature.nat_frames = 14;

specification.noisegroup(1).name = 'Accelerating, revving, vroom';
specification.noisegroup(1).percent = 50.0;
specification.noisegroup(1).type = 'stationary';

specification.noisegroup(2).name = 'Air brake';
specification.noisegroup(2).percent = 50.0;
specification.noisegroup(2).type = 'stationary';

s = specification;
end