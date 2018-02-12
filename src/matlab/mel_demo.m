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
load('D:\8-Workspace\train\spectrum\1+Air conditioning+m025wky1+I-nVcl1UdE4+dr3+mjvw0+sx113+-42.0+15.0.mat')
ratiomask_lin = ratiomask_dft(:,800);
b = filter_banks(16000, 512, 136, 0, 16000/2);
ratiomask_mel = (b * ratiomask_lin) ./ sum(b,2);
figure; plot(ratiomask_mel, 'r'); 

ratiomask_lin_recover = b' * ratiomask_mel;
figure; plot(ratiomask_lin_recover, 'r'); hold on; grid on;
plot(ratiomask_lin, 'b--');



%%
spectrum_mel = spectrum(:,800);
spectrum_mel = spectrum_mel ./ sum(b,2);
figure; plot(spectrum_mel, 'r');  hold on; grid on;
plot(b.' * spectrum_mel, 'b');

