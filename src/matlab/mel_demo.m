close all; clear all; clc;

b = filter_banks(16000, 512, 73, 0, 16000/2);

ratiomask_lin = rand(257,1);
ratiomask_mel = b * ratiomask_lin;
figure; plot(ratiomask_mel);

ratiomask_lin_1 = (b.' ./ (sum(b,2).')) * (ratiomask_mel);
ratiomask_lin_2 = pinv(b) * ratiomask_mel;

figure; plot(ratiomask_lin, 'r'); hold on; grid on;
plot(ratiomask_lin_1, 'b--');
plot(ratiomask_lin_2, 'k--');



%%
load('1+Air conditioning+m025wky1+I-nVcl1UdE4+dr3+mjvw0+sx113+-42.0+15.0.mat')
ratiomask_lin = ratiomask_dft(:,800);
b = filter_banks(16000, 512, 136, 0, 16000/2);
ratiomask_mel = (b * ratiomask_lin) ./ sum(b,2);
figure; plot(ratiomask_mel, 'r'); 

ratiomask_lin_recover = b' * ratiomask_mel;
figure; plot(ratiomask_lin_recover, 'r'); hold on; grid on;
plot(ratiomask_lin, 'b--');



%%
ratiomask_lin_noise = 1 - ratiomask_lin;
ratiomask_mel_noise = (b * ratiomask_lin_noise) ./ sum(b,2);
figure; hold on; grid on; 
plot(ratiomask_mel_noise, 'r');
plot(1 - ratiomask_mel, 'b--');


%% 
load('1+Air conditioning+m025wky1+I-nVcl1UdE4+dr3+mjvw0+sx113+-42.0+15.0.mat')
ratiomask_lin_bin = ratiomask_dft(:,800);
ratiomask_lin_bin(ratiomask_lin_bin > 0.5) = 1;
ratiomask_lin_bin(ratiomask_lin_bin <= 0.5) = 0;
plot(ratiomask_lin_bin); hold on; grid on;

ratiomask_mel_bin = (b * ratiomask_lin_bin) ./ sum(b,2);
ratiomask_lin_bin_recover = b' * ratiomask_mel_bin;
plot(ratiomask_lin_bin_recover)
