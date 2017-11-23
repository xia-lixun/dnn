% uses stft2.m from 

clear all
close all

%% load signal
% [x1,fs] = wavread('sound001.wav');
% [x2,fs] = wavread('sound002.wav');
[x1,fs] = wavread('original_speech2.wav');
[x2,fs] = wavread('original_noise2.wav');
M = min(length(x1),length(x2));
x1 = x1(1:M);
x2 = x2(1:M);
x = x1+x2;
x  = x(:).';
x1 = x1(:).';
x2 = x2(:).';


%% use stft2.m 
%    stft2( mixture, nFFT, hop, 0, wn); 
%     wav_singal = stft2( output.source_signal, sz(1), sz(2), 0, wn);
nFFT = 1024;
hop = 256;
win = sqrt( hann( nFFT, 'periodic')); % hann window
mixSpec = stft2(x, nFFT,hop,0,win);
x1Spec  = stft2(x1,nFFT,hop,0,win);
x2Spec  = stft2(x2,nFFT,hop,0,win);

bm = abs(x1Spec)>abs(x2Spec);
y1Spec = bm.*mixSpec;
y2Spec = (~bm).*mixSpec;

y  = stft2(mixSpec,nFFT,hop,0,win);
scale = 2; %stft2.m reconstruction always needs a scale factor of 2.
y = y*scale;
y1 = stft2(y1Spec,nFFT,hop,0,win)*scale;
y2 = stft2(y2Spec,nFFT,hop,0,win)*scale;
y = y(1:M);
y1 = y1(1:M);
y2 = y2(1:M);

10*log10(sum(abs(x-y).^2)/sum(x.^2))
figure;plot(x1);hold on;grid on;plot(y1,'r');plot(y1-x1,'k')
10*log10(sum(abs(x1-y1).^2)/sum(x1.^2))
figure;plot(x2);hold on;grid on;plot(y2,'r');plot(y2-x2,'k')
10*log10(sum(abs(x2-y2).^2)/sum(x2.^2))

