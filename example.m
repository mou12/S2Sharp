% An example illustrating how to use S2sharp using
% Sentinel-2 dataset simulated from Aviris.  For detail see 
% Sentinel-2 Sharpening Using a Reduced-Rank Method, M. Ulfarsson et al., 
% IEEE Transactions on Geoscience and Remote Sensing, 2019, 
% 
% https://www.researchgate.net/publication/332656193_Sentinel-2_Sharpening_Using_a_Reduced-Rank_Method

load Data/Aviris_cell_3.mat; 
r = 8; % subspace dimension / the rank
q  = [1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]; % lam_i = lam*qi
ni = 10;  % number of iterations / one can select fewer iterations (e.g. ni=2) which yields similar result
          % but is much quicker
lam = 1.8998e-04;
[ Xhat_im, output_S2 ]=S2sharp(Yim,'Xm_im',Xm_im,'r',r,'lambda',lam,'q',q,'CDiter',ni);

% Output
S2sharp_SRE = output_S2.SRE{end}([1,5,6,7,9:12]);
S2sharp_SAM = output_S2.SAMm(end);
S2sharp_aSRE= mean(S2sharp_SRE );    
S2sharp_RMSE = output_S2.RMSE(end);
S2sharp_aSSIM = output_S2.aSSIM(end);
S2sharp_ERGAS_60m=output_S2.ERGAS_60m(end);
S2sharp_ERGAS_20m=output_S2.ERGAS_20m(end);
S2sharp_time = output_S2.Time;

disp(['S2sharp: Best lambda=' num2str(lam(end))]);
disp(['S2sharp: SAM=' num2str(S2sharp_SAM)])
disp(['Average SRE = ' num2str(S2sharp_aSRE)]);
disp(['S2sharp aSSIM = ' num2str(S2sharp_aSSIM)]);
disp(['S2sharp RMSE = ' num2str(S2sharp_RMSE)]);
disp(['S2sharp time = ' num2str(S2sharp_time)]);
disp(['S2sharp: SRE:'])
disp(['B1    B5    B6    B7    B8a   B9    B11   B12'])
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n', S2sharp_SRE)


