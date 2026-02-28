% export_reference.m
% Runs S2sharp for one iteration and saves intermediate values
% to Data/matlab_reference.mat for validating the Python port.
%
% Usage: Run this script from the project root directory in MATLAB.
%        Requires the Data/Aviris_cell_3.mat dataset.

addpath manopt
p1 = pwd;
cd('manopt');
importmanopt
cd(p1)

load Data/Aviris_cell_3.mat;

% Parameters (matching example.m)
r = 8;
q = [1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]';
lambda = 1.8998e-04;
CDiter = 10;
tolgradnorm = 0.1;

% Dimensions
L = length(Yim);
Yim = reshape(Yim, L, 1);
for i = 1:L, Yim{i} = double(Yim{i}); end
[nl, nc] = size(Yim{2});
n = nl * nc;

% Normalize data
[Yim2, av] = normaliseData(Yim);

% Band parameters
d = [6 1 1 1 2 2 2 1 2 6 2 2]';
mtf = [.32 .26 .28 .24 .38 .34 .34 .26 .33 .26 .22 .23];
sdf = d .* sqrt(-2 * log(mtf) / pi^2)';
sdf(d == 1) = 0;
limsub = 2;
dx = 12;
dy = 12;

% === Checkpoint 1: FBM2 (subspace blur kernels) ===
FBM2 = createConvKernelSubspace(sdf, nl, nc, L, dx, dy);

% === Checkpoint 2: Ylim (upsampled bands) ===
Ylim = zeros(nl, nc, L);
for i = 1:L
    Ylim(:,:,i) = imresize(Yim2{i}, d(i));
end

% === Checkpoint 3: F_init (initial subspace from SVD) ===
Y2im = real(ifft2(fft2(Ylim) .* FBM2));
Y2tr = Y2im(limsub+1:end-limsub, limsub+1:end-limsub, :);
Y2n = reshape(Y2tr, [(nl-4)*(nc-4), L]);
[F_init, ~, ~] = svd(Y2n', 'econ');
F_init = F_init(:, 1:r);

% === Checkpoint 4: Y, Mask (observed data and subsampling mask) ===
FBM = createConvKernel(sdf, d, nl, nc, L, dx, dy);
[Y, M, F] = initialization(Yim2, sdf, nl, nc, L, dx, dy, d, limsub, r);
Mask = reshape(M, [n, L])';

% === Checkpoint 5: W (adaptive weights) ===
sigmas = 1;
W = computeWeights(Y, d, sigmas, nl);

% === Run iterations and save checkpoints ===
Z = zeros(r, n);
[FDH, FDV, FDHC, FDVC] = createDiffkernels(nl, nc, r);
Whalf = W.^(1/2);

% Iteration 1: Z-step
[Z, ~, ~] = Zstep(Y, FBM, F, lambda, nl, nc, Z, Mask, q, FDH, FDV, FDHC, FDVC, W, Whalf, tolgradnorm);
Z_iter1 = Z;

% Iteration 1: F-step
F1 = Fstep(F, Z, Y, FBM, nl, nc, Mask);
F_iter1 = F1;
F = F1;

% Continue for remaining iterations
for jCD = 2:CDiter
    [Z, ~, ~] = Zstep(Y, FBM, F, lambda, nl, nc, Z, Mask, q, FDH, FDV, FDHC, FDVC, W, Whalf, tolgradnorm);
    F1 = Fstep(F, Z, Y, FBM, nl, nc, Mask);
    F = F1;
end

% === Final metrics ===
Xhat_im = conv2im(F*Z, nl, nc, L);
[SAMm_final, SAMm_2m_final, SRE_final, RMSE_final, SSIM_final, aSSIM_final, ...
    ERGAS_20m_final, ERGAS_60m_final] = evaluate_performance(Xm_im, Xhat_im, nl, nc, L, limsub, d, av);

% === Save all reference values ===
save('Data/matlab_reference.mat', ...
    'FBM2', 'Ylim', 'F_init', 'Y', 'Mask', 'W', ...
    'Z_iter1', 'F_iter1', ...
    'SAMm_final', 'SAMm_2m_final', 'SRE_final', 'RMSE_final', ...
    'SSIM_final', 'aSSIM_final', 'ERGAS_20m_final', 'ERGAS_60m_final', ...
    'av', 'sdf', 'd', 'nl', 'nc', 'L', 'r', 'lambda', 'q', ...
    '-v7.3');

fprintf('Reference values saved to Data/matlab_reference.mat\n');
fprintf('Final metrics:\n');
fprintf('  SAM:       %.6f\n', SAMm_final);
fprintf('  aSSIM:     %.6f\n', aSSIM_final);
fprintf('  RMSE:      %.6f\n', RMSE_final);
fprintf('  ERGAS_20m: %.6f\n', ERGAS_20m_final);
fprintf('  ERGAS_60m: %.6f\n', ERGAS_60m_final);

% === Local helper functions (copied from S2sharp.m) ===

function [Yim, av] = normaliseData(Yim)
    if iscell(Yim)
        nb = length(Yim);
        for i=1:nb
            av(i,1) = mean2(Yim{i}.^2);
            Yim{i,1} = sqrt(Yim{i}.^2/av(i,1));
        end
    else
        nb = size(Yim,3);
        for i=1:nb
            av(i,1) = mean2(Yim(:,:,i).^2);
            Yim(:,:,i) = sqrt(Yim(:,:,i).^2/av(i,1));
        end
    end
end

function FBM = createConvKernel(sdf, d, nl, nc, L, dx, dy)
    middlel = ((nl)/2);
    middlec = ((nc)/2);
    B = zeros(nl,nc,L);
    FBM = zeros(nl,nc,L);
    for i=1:L
        if d(i) > 1
            h = fspecial('gaussian',[dx,dy],sdf(i));
            B((middlel-dy/2+1:middlel+dy/2)-d(i)/2+1,(middlec-dx/2+1:middlec+dx/2)-d(i)/2+1,i) = h;
            B(:,:,i) = fftshift(B(:,:,i));
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM(:,:,i) = fft2(B(:,:,i));
        else
            B(1,1,i) = 1;
            FBM(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function FBM2 = createConvKernelSubspace(sdf, nl, nc, L, dx, dy)
    middlel = round((nl+1)/2);
    middlec = round((nc+1)/2);
    dx = dx+1;
    dy = dy+1;
    B = zeros(nl,nc,L);
    FBM2 = zeros(nl,nc,L);
    s2 = max(sdf);
    for i=1:L
        if sdf(i) < s2
            h = fspecial('gaussian',[dx,dy],sqrt(s2^2-sdf(i)^2));
            B(middlel-(dy-1)/2:middlel+(dy-1)/2,middlec-(dx-1)/2:middlec+(dx-1)/2,i) = h;
            B(:,:,i) = fftshift(B(:,:,i));
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM2(:,:,i) = fft2(B(:,:,i));
        else
            B(1,1,i) = 1;
            FBM2(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function [Y, M, F] = initialization(Yim2, sdf, nl, nc, L, dx, dy, d, limsub, r)
    FBM2 = createConvKernelSubspace(sdf, nl, nc, L, dx, dy);
    for i=1:L
        Ylim(:,:,i) = imresize(Yim2{i}, d(i));
    end
    Y2im = real(ifft2(fft2(Ylim) .* FBM2));
    Y2tr = Y2im(limsub+1:end-limsub, limsub+1:end-limsub, :);
    Y2n = reshape(Y2tr, [(nl-4)*(nc-4), L]);
    [F, ~, ~] = svd(Y2n', 'econ');
    F = F(:, 1:r);
    [M, Y] = createSubsampling(Yim2, d, nl, nc, L);
end

function [M, Y] = createSubsampling(Yim, d, nl, nc, L)
    M = zeros(nl, nc, L);
    indexes = cell([L 1]);
    for i=1:L
        im = ones(floor(nl/d(i)), floor(nc/d(i)));
        maux = zeros(d(i));
        maux(1,1) = 1;
        M(:,:,i) = kron(im, maux);
        indexes{i} = find(M(:,:,i) == 1);
        Y(i, indexes{i}) = conv2mat(Yim{i}, nl/d(i), nc/d(i), 1);
    end
end

function [FDH, FDV, FDHC, FDVC] = createDiffkernels(nl, nc, r)
    dh = zeros(nl, nc);
    dh(1,1) = 1;
    dh(1,nc) = -1;
    dv = zeros(nl, nc);
    dv(1,1) = 1;
    dv(nl,1) = -1;
    FDH = repmat(fft2(dh), 1, 1, r);
    FDV = repmat(fft2(dv), 1, 1, r);
    FDHC = conj(FDH);
    FDVC = conj(FDV);
end

function X = conv2mat(X, nl, nc, L)
    if ndims(X) == 3
        [nl, nc, L] = size(X);
        X = reshape(X, nl*nc, L)';
    elseif ndims(squeeze(X)) == 2
        L = 1;
        [nl, nc] = size(X);
        X = reshape(X, nl*nc, L)';
    end
end

function X = conv2im(X, nl, nc, L)
    if size(X,2) == 1
        X = conv2mat(X, nl, nc, L);
    end
    if nargin == 2
        [L, n] = size(X);
        if n == 1
            X = conv2mat(X, nl, nc, L);
        end
        nc = n/nl;
    end
    X = reshape(X', nl, nc, L);
end

function X = ConvCM(X, FKM, nl, nc, L)
    if nargin == 3
        [L, n] = size(X);
        nc = n/nl;
    end
    X = conv2mat(real(ifft2(fft2(conv2im(X,nl,nc,L)) .* FKM)));
end

function W = computeWeights(Y, d, sigmas, nl)
    hr_bands = d == 1;
    hr_bands = find(hr_bands)';
    for i = hr_bands
        grad(:,:,i) = imgradient(conv2im(Y(i,:), nl), 'intermediate').^2;
    end
    grad = sqrt(max(grad, [], 3));
    grad = grad / quantile(grad(:), 0.95);
    Wim = exp(-grad.^2/2/sigmas^2);
    Wim(Wim < 0.5) = 0.5;
    W = conv2mat(Wim, nl);
end

function [Z, xcost, options] = Zstep(Y, FBM, F, tau, nl, nc, Z, Mask, q, FDH, FDV, FDHC, FDVC, W, Whalf, tolgradnorm)
    r = size(F, 2);
    n = nl * nc;
    UBTMTy = F' * ConvCM(Y, conj(FBM), nl);
    [Z] = CG(Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q, FDH, FDV, FDHC, FDVC, W);
    xcost = 1;
    options = [];
end

function F1 = Fstep(F, Z, Y, FBM, nl, nc, Mask)
    F0 = F;
    BTXhat = ConvCM(F0*Z, FBM, nl);
    MBTXhat = Mask .* BTXhat;
    [L, r] = size(F);
    for ii = 1:L
        MBZT(:,:,ii) = repmat(Mask(ii,:), [r,1]) .* ConvCM(Z, repmat(FBM(:,:,ii), [1,1,r]), nl);
        A(:,:,ii) = MBZT(:,:,ii) * MBZT(:,:,ii)';
        ZBMTy(:,ii) = MBZT(:,:,ii) * Y(ii,:)';
    end
    ZBYT = ZBMTy';
    manifold = stiefelfactory(L, r, 1);
    problem.M = manifold;
    problem.cost = @(F) costF(F, MBZT, Y);
    problem.egrad = @(F) egrad(F, A, ZBYT);
    warning('off', 'manopt:getHessian:approx')
    options.tolgradnorm = 1e-2;
    options.verbosity = 0;
    [F1, ~, ~, ~] = trustregions(problem, F0, options);
end

function [Ju] = costF(F, MBZT, Y)
    L = size(F, 1);
    Ju = 0;
    for i = 1:L
        fi = F(i,:)';
        yi = Y(i,:)';
        Ju = Ju + 0.5 * norm(MBZT(:,:,i)'*fi - yi, 'fro')^2;
    end
end

function [Du] = egrad(F, A, ZBYT)
    p = size(A, 3);
    Du = 0 * F;
    for ii = 1:p
        Du(ii,:) = F(ii,:) * A(:,:,ii)' - ZBYT(ii,:);
    end
end

function [J, gradJ, AtAg] = grad_cost_G(Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q, FDH, FDV, FDHC, FDVC, W)
    X = F * Z;
    BX = ConvCM(X, FBM, nl);
    HtHBX = Mask .* BX;
    ZH = ConvCM(Z, FDHC, nl);
    Zv = ConvCM(Z, FDVC, nl);
    ZHW = ZH .* W;
    ZVW = Zv .* W;
    grad_pen = ConvCM(ZHW, FDH, nl) + ConvCM(ZVW, FDV, nl);
    AtAg = F' * ConvCM(HtHBX, conj(FBM), nl) + 2*tau*(q*ones(1,nl*nc)) .* grad_pen;
    gradJ = AtAg - UBTMTy;
    J = 1/2 * sum(sum(Z .* AtAg)) - sum(sum(Z .* UBTMTy));
end

function [Z] = CG(Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q, FDH, FDV, FDHC, FDVC, W)
    maxiter = 1000;
    tolgradnorm = 0.1;
    [cost, grad] = grad_cost_G(Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q, FDH, FDV, FDHC, FDVC, W);
    gradnorm = norm(grad(:));
    iter = 0;
    res = -grad;
    while (gradnorm > tolgradnorm & iter < maxiter)
        iter = iter + 1;
        if (iter == 1)
            desc_dir = res;
        else
            beta = (res(:).' * res(:)) / (old_res(:).' * old_res(:));
            desc_dir = res + beta * desc_dir;
        end
        [~, ~, AtAp] = grad_cost_G(desc_dir, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q, FDH, FDV, FDHC, FDVC, W);
        alpha = (res(:).' * res(:)) / (desc_dir(:).' * AtAp(:));
        Z1 = Z + alpha * desc_dir;
        old_res = res;
        res = res - alpha * AtAp;
        gradnorm = norm(res(:));
        Z = Z1;
    end
end

function [Yim] = unnormaliseData(Yim, av)
    if iscell(Yim)
        nb = length(Yim);
        for i=1:nb
            Yim{i,1} = sqrt(Yim{i}.^2*av(i,1));
        end
    else
        nb = size(Yim,3);
        for i=1:nb
            Yim(:,:,i) = sqrt(Yim(:,:,i).^2*av(i,1));
        end
    end
end

function [SAMm, SAMm_2m, SRE, RMSE, SSIM_index, aSSIM, ERGAS_20m, ERGAS_60m] = evaluate_performance(Xm_im, Xhat_im, nl, nc, L, limsub, d, av)
    Xhat_im = Xhat_im(limsub+1:end-limsub, limsub+1:end-limsub, :);
    Xhat_im = unnormaliseData(Xhat_im, av);
    Xhat = reshape(Xhat_im, [(nl-4)*(nc-4), L]);
    Xm_im = Xm_im(limsub+1:end-limsub, limsub+1:end-limsub, :);
    if (size(Xm_im, 3) == 6)
        ind = find(d == 2);
        SAMm = SAM(Xm_im, Xhat_im(:,:,ind));
        SAMm_2m = SAMm;
        X = conv2mat(Xm_im);
        Xhat = conv2mat(Xhat_im);
        for i = 1:6
            SRE(i,1) = 10*log10(sum(X(i,:).^2) / sum((Xhat(ind(i),:)-X(i,:)).^2));
            SSIM_index(i,1) = ssim(Xm_im(:,:,i), Xhat_im(:,:,ind(i)));
        end
        aSSIM = mean(SSIM_index);
        ERGAS_20m = ERGAS(Xm_im, Xhat_im(:,:,ind), 2);
        ERGAS_60m = nan;
        RMSE = norm(X - Xhat(ind,:), 'fro') / size(X, 2);
    else
        ind = find(d==2 | d==6);
        SAMm = SAM(Xm_im(:,:,ind), Xhat_im(:,:,ind));
        ind2 = find(d == 2);
        SAMm_2m = SAM(Xm_im(:,:,ind2), Xhat_im(:,:,ind2));
        X = conv2mat(Xm_im);
        Xhat = conv2mat(Xhat_im);
        for i = 1:L
            SRE(i,1) = 10*log10(sum(X(i,:).^2) / sum((Xhat(i,:)-X(i,:)).^2));
            SSIM_index(i,1) = ssim(Xm_im(:,:,i), Xhat_im(:,:,i));
        end
        aSSIM = mean(SSIM_index(ind));
        ERGAS_20m = ERGAS(Xm_im(:,:,ind), Xhat_im(:,:,ind), 2);
        ERGAS_60m = ERGAS(Xm_im(:,:,ind2), Xhat_im(:,:,ind2), 6);
        RMSE = norm(X(ind,:) - Xhat(ind,:), 'fro') / size(X, 2);
    end
end

function [SAM_index, SAM_map] = SAM(I1, I2)
    [M, N, ~] = size(I2);
    prod_scal = dot(I1, I2, 3);
    norm_orig = dot(I1, I1, 3);
    norm_fusa = dot(I2, I2, 3);
    prod_norm = sqrt(norm_orig .* norm_fusa);
    prod_map = prod_norm;
    prod_map(prod_map == 0) = eps;
    SAM_map = acos(prod_scal ./ prod_map);
    prod_scal = reshape(prod_scal, M*N, 1);
    prod_norm = reshape(prod_norm, M*N, 1);
    z = find(prod_norm == 0);
    prod_scal(z) = [];
    prod_norm(z) = [];
    angolo = sum(sum(acos(prod_scal ./ prod_norm))) / (size(prod_norm, 1));
    SAM_index = real(angolo) * 180/pi;
end

function ERGAS_index = ERGAS(I1, I2, ratio)
    I1 = double(I1);
    I2 = double(I2);
    Err = I1 - I2;
    ERGAS_index = 0;
    for iLR = 1:size(Err, 3)
        ERGAS_index = ERGAS_index + mean2(Err(:,:,iLR).^2) / (mean2((I1(:,:,iLR))))^2;
    end
    ERGAS_index = (100/ratio) * sqrt((1/size(Err,3)) * ERGAS_index);
end
