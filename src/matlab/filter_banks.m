function f = filter_banks(fs, nfft, filt_num, hz_low, hz_high)
% f = filter_banks(16000, 512, 26, 0, 16000/2)
    if hz_high > fs / 2
        f = zeros(filt_num,nfft/2+1);
    else
        mel_low = hz_to_mel(hz_low);
        mel_high = hz_to_mel(hz_high);
        mel_points = linspace(mel_low, mel_high, filt_num+2);
        hz_points = mel_to_hz(mel_points);
        
        %round frequencies to nearest fft bins
        b = floor((hz_points/fs) * (nfft+1));
        f = zeros(filt_num,nfft/2+1);
        for i = 1:filt_num
            for j = b(i):b(i+1)
                f(i,j+1) = (j - b(i)) / (b(i+1) - b(i));
            end
            for j = b(i+1):b(i+2)
                f(i,j+1) = (b(i+2) - j) / (b(i+2) - b(i+1));
            end
        end
    end
    f(isnan(sum(f,2)),:) = [];
end



function mel = hz_to_mel(hz)
    mel = 2595 * log10(1 + hz * 1.0 / 700);
end


function hz = mel_to_hz(mel)
    hz = 700 * (10 .^ (mel * 1.0 / 2595) - 1);
end