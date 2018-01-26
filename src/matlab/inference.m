function inference(s, model_path, wav_path)
    
    model = loadmodel(model_path);
    todo =  dir([wav_path '/**/*.wav']);

    nfft = s.feature.frame_length;
    hop = s.feature.hop_length;
    win = sqrt(hann(nfft,'periodic'));
    
    for i = 1:length(todo)
        
        path = fullfile(todo(i).folder, todo(i).name);
        [x,fs] = audioread(path);
        assert(fs == s.sample_rate)
        
        spect = stft2(x.', nfft, hop, 0, win);
        tensor = sliding((abs(spect) - model.stat.mu)./model.stat.std, (s.feature.context_span-1)/2, s.feature.nat_frames);
        bm_hat = feed_forward(model, tensor);
        
        % reconstruct based on bm estimate
        model_reconstruct = bm_hat .* spect;
        scale = 2;
        speech_recovered = stft2(model_reconstruct, nfft, hop, 0, win) * scale;
        audiowrite([path(1:end-4) '-processed.wav'], speech_recovered, s.sample_rate, 'BitsPerSample', 32);
    end
    
end




function m = loadmodel(model_path)
    
    load(model_path, 'mu','stdev','W1','b1','W2','b2','W3','b3','W4','b4');
    model.nn(1).w = W1.';
    model.nn(1).b = b1.';
    model.nn(2).w = W2.';
    model.nn(2).b = b2.';
    model.nn(3).w = W3.';
    model.nn(3).b = b3.';
    model.nn(4).w = W4.';
    model.nn(4).b = b4.';
    model.stat.mu = mu;
    model.stat.std = stdev;
    m = model;
end




function y = feed_forward(model, x)
    
    layers = length(model.nn);
    act = sigmoid(model.nn(1).w * x + model.nn(1).b);
    for j = 2:layers-1
        act = sigmoid(model.nn(j).w * act + model.nn(j).b);
    end
    y = sigmoid(model.nn(layers).w * act + model.nn(layers).b);
end


function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
