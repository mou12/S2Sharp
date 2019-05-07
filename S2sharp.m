function [Xhat_im , output ] = S2sharp(Yim,varargin)
% Usage [ Xhat_im , output ] = S2sharp(Yim,varargin)
%
% When using this code please refer to
%
% Sentinel-2 Sharpening Using a Reduced-Rank Method, M.O. Ulfarsson et al.
% IEEE Transactions on Geoscience and Remote Sensing, 2019.
%
% This method sharpens bands the 60m bands B1,B9 and the 20m bands
% B4,B5,B6,B7,B8a,B11,B12 to 10m resolution.  We assume that
% the noisy satellite images yi, i=1,...,L are related to the full
% resolution target images by
%
%   yi = Mi*Bi*xi + ni, i=1,...,L 
% 
% where Mi is a downsampling operator, Bi is a circulant blurring matrix,
% and ni is noise.  The method solves
%        min (1/2) sum_{i=1}^L || y_i - Mi*Bi*G*fi ||^2  + sum_{j=1}^r lam_j * phi(gj)
%        F, G
% where phi is a regularizer function.
% The function returns Xhat=G*F'. 
%
%
% Input:
%         Yim : 1x12 cell array containing the observed images for each of 
%               the nb bands of the Sentinel 2 sensor.
%       CDiter: Number of cyclic descent iterations. 
%               CDiter=10 is the default.
%            r: The subspace dimension, r=7 is the default.
%       lambda: The regularization parameter, lambda=0.005 is the 
%               default.
%        Xm_im: nl x nc x nb 3D matrix containing the true (10m resolution)
%            q: penalty weights, if r=7 then q= [1, 1.5, 4, 8, 15, 15, 20 ]'
%               is the default otherwise the default is q=ones(p,1). Note
%               that lam_i = lam*q_i
%           X0: Initial value for X = G * F'
%   Gstep_only: If Gstep_only=1 then perform the G-step (once). Assuming that F is fixed
%          GCV: If GCV=1 then the GCV value is computed.
% Output:   output is a structure containing the following fields
%    Xhat_im: estimated image (3D) at high resolution (10m) for each 
%             spectral channel
%       SAMm: mean SAM for the 60m and 20m bands (empty if Xm_im is not
%             available)
%    SAMm_2m: mean SAM for the 20m bands (empty if Xm_im is not available)
%        SRE: signal to reconstruction error for all the 12 bands
%             (empty if Xm_im is not available)
%        GCVscore: Contains the GCV score if it was GCV=1 otherwise GCV is
%        empty.
%    ERGAS_20m: ERGAS score for the 20 m bands
%    ERGAS_60m: ERGAS score for the 60 m bands
%        aSSIM: average Structural Similarity Index 
%         RMSE: Root mean squared error
%         Time: computational time
% Magnus O. Ulfarsson, September 2017.  
% Acknowledgement:
%
% S2sharp uses some code from
%
% C. Lanaras, J. Bioucas-Dias, E. Baltsavias, and K. Schindler, “Superresolution
% of multispectral multiresolution images from a single sensor,”
% in IEEE Conference on Computer Vision and Pattern Recognition
% Workshops (CVPRW), 2017.
%
% https://github.com/lanha/SupReME
%
% The Manopt software
% https://www.manopt.org/
%
    % Import the manopt optimizer
    addpath manopt
    p1=pwd;
    cd('manopt');
    importmanopt
    cd(p1)
    % initialization
    CDiter=10;
    r=7;
    lambda=0.005;
    Xm_im='';
    X0 = '';
    tolgradnorm = 0.1;
    if(r==7)
        q = [1, 1.5, 4, 8, 15, 15, 20 ]';
    else
        q = ones(r,1);
    end
    Gstep_only=0;
    GCV = 0;
    output = struct('SAMm',[],'SAMm_2m',[],'SRE',[], 'GCVscore',[], 'ERGAS_20m', [], ...
        'ERGAS_60m', [], 'SSIM', [], 'aSSIM', [], 'RMSE', [], 'Time', []);
    for i=1:2:(length(varargin)-1)
        switch varargin{i}
            case 'CDiter'
                CDiter=varargin{i+1};
            case 'r'
                r=varargin{i+1};
            case 'lambda'
                lambda=varargin{i+1};
            case 'Xm_im'
                Xm_im=varargin{i+1};
            case 'q'
                q=varargin{i+1};
            case 'X0'
                X0 = varargin{i+1};
            case 'tolgradnorm'
                tolgradnorm = varargin{i+1};
            case 'Gstep_only'
                Gstep_only = varargin{i+1};
            case 'GCV'
                GCV = varargin{i+1};
        end
    end
    tic;
    if(length(q)~=r), error('The length of q has to match r'); end
    q = q(:);
    % dimensions of the inputs
    L=length(Yim);
    Yim=reshape(Yim,L,1);
    for i=1:L, Yim{i}=double(Yim{i}); end
    [nl,nc] = size(Yim{2});
    n = nl*nc;
    [Yim2, av] = normaliseData(Yim);
    % Sequence of bands
    % [B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12]
    % subsampling factors (in pixels)
    d = [6 1 1 1 2 2 2 1 2 6 2 2]'; % 
    % convolution  operators (Gaussian convolution filters), taken from ref [5]
    mtf = [ .32 .26 .28 .24 .38 .34 .34 .26 .33 .26 .22 .23];
    sdf = d.*sqrt(-2*log(mtf)/pi^2)';
    sdf(d==1) = 0;
    limsub = 2;
    dx = 12;
    dy = 12;
    FBM = createConvKernel(sdf,d,nl,nc,L,dx,dy);
    [Y,M,F]=initialization(Yim2,sdf,nl,nc,L,dx,dy,d,limsub,r);
    Mask=reshape(M,[n,L])';
    % CD
    if isempty(X0)
        Z = zeros(r,n); 
    else
        [X0, ~] = normaliseData(X0);
        X0 = reshape(X0,[n,L])';
        [F,D,V]=svd(X0,'econ');
        F = F(:,1:r);
        Z = D(1:r,1:r)*V(:,1:r)';
    end
    [FDH,FDV,FDHC,FDVC] = createDiffkernels(nl,nc,r);
    sigmas = 1;
    W = computeWeights(Y,d,sigmas,nl);
    Whalf=W.^(1/2);
    if( GCV == 1), Gstep_only=1; end
    if( Gstep_only ~= 0), CDiter=1; end
    for jCD=1:CDiter
       [Z,Jcost(jCD),options]=Zstep(Y,FBM,F,lambda,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm);              
       if(Gstep_only==0) 
           F1=Fstep(F,Z,Y,FBM,nl,nc,Mask);  
           F=F1;
       end
       if( GCV==1 )
            Ynoise = ( abs(Y) > 0 ) .* randn( size(Y) );
            [Znoise]=Zstep(Ynoise,FBM,F,lambda,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm);
            HtHBXnoise = Mask.*ConvCM(F*Znoise,FBM,nl);
            Ynoise = Ynoise([1,5,6,7,9:12],:); 
            HtHBXnoise = HtHBXnoise([1,5,6,7,9:12],:);
            den = trace(Ynoise*(Ynoise - HtHBXnoise)');           
            HtHBX=Mask.*ConvCM(F*Z,FBM,nl); 
            num = norm( Y([1,5,6,7,9:12],:) - HtHBX([1,5,6,7,9:12],:) , 'fro')^2;         
            output.GCVscore = num / den ; 
       end
       output.Time=toc;
       if(~isempty(Xm_im))
           Xhat_im = conv2im(F*Z,nl,nc,L);
           [output.SAMm(jCD), output.SAMm_2m(jCD), output.SRE{jCD}, output.RMSE(jCD), ...
               output.SSIM{jCD}, output.aSSIM(jCD),...
               output.ERGAS_20m(jCD), output.ERGAS_60m(jCD)] ...
               = evaluate_performance(Xm_im,Xhat_im,nl,nc,L,limsub,d,av);
       end
    end
    Xhat_im = conv2im(F*Z,nl,nc,L);
    Xhat_im = Xhat_im(limsub+1:end-limsub,limsub+1:end-limsub,:);
    Xhat_im = unnormaliseData(Xhat_im,av);
end

function [Y,M,F]=initialization(Yim2,sdf,nl,nc,L,dx,dy,d,limsub,r)
    FBM2 = createConvKernelSubspace(sdf,nl,nc,L,dx,dy);
    for i=1:L
        Ylim(:,:,i) = imresize(Yim2{i},d(i));
    end
    Y2im=real(ifft2(fft2(Ylim).*FBM2));
    Y2tr=Y2im(limsub+1:end-limsub,limsub+1:end-limsub,:);
    Y2n = reshape(Y2tr,[(nl-4)*(nc-4),L]); 
    [F,D,P] = svd(Y2n','econ');
    F=F(:,1:r);
    [M, Y] = createSubsampling(Yim2,d,nl,nc,L);
end


function [Z, xcost,options]=Zstep(Y,FBM,F,tau,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm)
    r = size(F,2);
    n = nl*nc;     
    UBTMTy=F'*ConvCM(Y,conj(FBM),nl); 
    [Z] = CG(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
    xcost=1;
    options=[];    
end      

function F1=Fstep(F,Z,Y,FBM,nl,nc,Mask)
     F0=F;%   U; % initialization
     BTXhat =  ConvCM(F0*Z,FBM,nl);
     MBTXhat=Mask.*BTXhat;
     [L,r]=size(F);
     for ii=1:L
        MBZT(:,:,ii)=repmat(Mask(ii,:),[r,1]).*ConvCM(Z,repmat(FBM(:,:,ii),[1,1,r]),nl);
        A(:,:,ii)=MBZT(:,:,ii)*MBZT(:,:,ii)';
        ZBMTy(:,ii)=MBZT(:,:,ii)*Y(ii,:)';
     end
     ZBYT=ZBMTy';%    BTY*Z';
     manifold = stiefelfactory(L,r,1); %euclideanfactory(L,r); 
     problem.M = manifold;
     problem.cost  = @(F) costF(F,MBZT,Y); 
     problem.egrad = @(F) egrad(F,A,ZBYT);  
     warning('off', 'manopt:getHessian:approx') 
     options.tolgradnorm = 1e-2;
     options.verbosity=0;
     [F1, xcost, info, options] = trustregions(problem,F0,options);
end

% Cost functions

function [Ju]=costF(F,MBZT,Y)
    L=size(F,1);
    Ju=0;
    for i=1:L,
        fi=F(i,:)';
        yi=Y(i,:)';
        Ju=Ju+0.5*norm(MBZT(:,:,i)'*fi-yi,'fro')^2;
    end
end

function [Du]=egrad(F,A,ZBYT)
    p=size(A,3);
    Du=0*F;
    for ii=1:p
        Du(ii,:)=F(ii,:)*A(:,:,ii)'-ZBYT(ii,:);
    end
end

function [SAMm, SAMm_2m, SRE, RMSE, SSIM_index, aSSIM, ERGAS_20m, ERGAS_60m] = evaluate_performance(Xm_im,Xhat_im,nl,nc,L,limsub,d,av)
    Xhat_im = Xhat_im(limsub+1:end-limsub,limsub+1:end-limsub,:);
    Xhat_im = unnormaliseData(Xhat_im,av);
    Xhat=reshape(Xhat_im,[(nl-4)*(nc-4),L]);
    % Xm_im is the ground truth image
    Xm_im = Xm_im(limsub+1:end-limsub,limsub+1:end-limsub,:);         
    if ( size(Xm_im,3) == 6 ) % Reduced Resolution
        ind = find( d==2 );
        SAMm=SAM(Xm_im,Xhat_im(:,:,ind));
        SAMm_2m=SAMm;
        X = conv2mat(Xm_im); 
        Xhat = conv2mat(Xhat_im);
        % SRE - signal to reconstrution error
        for i=1:6
            SRE(i,1) = 10*log10(sum(X(i,:).^2)/ sum((Xhat(ind(i),:)-X(i,:)).^2));
            SSIM_index(i,1) = ssim(Xm_im(:,:,i),Xhat_im(:,:,ind(i)));
        end
        aSSIM=mean(SSIM_index);
        ERGAS_20m = ERGAS(Xm_im,Xhat_im(:,:,ind),2);
        ERGAS_60m = nan;
        RMSE = norm(X - Xhat(ind,:),'fro') / size(X,2);
    else    
        ind=find(d==2 | d==6);
        SAMm=SAM(Xm_im(:,:,ind),Xhat_im(:,:,ind));
        ind2=find(d==2);
        SAMm_2m=SAM(Xm_im(:,:,ind2),Xhat_im(:,:,ind2));
        ind6=find(d==6);
        X = conv2mat(Xm_im); 
        Xhat = conv2mat(Xhat_im);
        % SRE - signal to reconstrution error
        for i=1:L
            SRE(i,1) = 10*log10(sum(X(i,:).^2)/ sum((Xhat(i,:)-X(i,:)).^2));
            SSIM_index(i,1) = ssim(Xm_im(:,:,i),Xhat_im(:,:,i));
        end
        aSSIM=mean(SSIM_index(ind));
        ERGAS_20m = ERGAS(Xm_im(:,:,ind),Xhat_im(:,:,ind),2);
        ERGAS_60m = ERGAS(Xm_im(:,:,ind2),Xhat_im(:,:,ind2),6);
        RMSE = norm(X(ind,:) - Xhat(ind,:),'fro') / size(X,2);
    end
end



%%% AUXILILARY FUNCTIONS

function [FDH,FDV,FDHC,FDVC] = createDiffkernels(nl,nc,r)
    dh = zeros(nl,nc);
    dh(1,1) = 1;
    dh(1,nc) = -1;
    dv = zeros(nl,nc);
    dv(1,1) = 1;
    dv(nl,1) = -1;
    FDH = repmat(fft2(dh),1,1,r);
    FDV = repmat(fft2(dv),1,1,r);
    FDHC = conj(FDH);
    FDVC = conj(FDV);
end


function z=vec(Z)
    z=Z(:);
end

function [Yim, av] = normaliseData(Yim)
    % Normalize each cell to unit power
    if iscell(Yim)
        % mean squared power = 1
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

function FBM = createConvKernel(sdf,d,nl,nc,L,dx,dy)
    %--------------------------------------------------------------------------
    %   Build convolution kernels
    %--------------------------------------------------------------------------
    %
    middlel=((nl)/2);
    middlec=((nc)/2);
    % kernel filters expanded to size [nl,nc]
    B = zeros(nl,nc,L);
    % fft2 of kernels
    FBM = zeros(nl,nc,L);
    for i=1:L
        if d(i) > 1
            h = fspecial('gaussian',[dx,dy],sdf(i));
            B((middlel-dy/2+1:middlel+dy/2)-d(i)/2+1,(middlec-dx/2+1:middlec+dx/2)-d(i)/2+1,i) = h; %run
           
            B(:,:,i)= fftshift(B(:,:,i));
            
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM(:,:,i) = fft2(B(:,:,i));
        else
            B(1,1,i) = 1;
            FBM(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function FBM2 = createConvKernelSubspace(sdf,nl,nc,L,dx,dy)

    %--------------------------------------------------------------------------
    %   Build convolution kernels FOR SUBSPACE!!!!
    %--------------------------------------------------------------------------
    %
    middlel=round((nl+1)/2);
    middlec=round((nc+1)/2);

    dx = dx+1;
    dy = dy+1;

    % kernel filters expanded to size [nl,nc]
    B = zeros(nl,nc,L);
    % fft2 of kernels
    FBM2 = zeros(nl,nc,L);

    s2 = max(sdf);
    for i=1:L
        if sdf(i) < s2
            h = fspecial('gaussian',[dx,dy],sqrt(s2^2-sdf(i)^2));
            B(middlel-(dy-1)/2:middlel+(dy-1)/2,middlec-(dx-1)/2:middlec+(dx-1)/2,i) = h;
    %       B((middlel-dy/2+1:middlel+dy/2),(middlec-dx/2+1:middlec+dx/2),i) = h;
            %circularly center
            B(:,:,i)= fftshift(B(:,:,i));
            % normalize
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM2(:,:,i) = fft2(B(:,:,i));
        else
            % unit impulse
            B(1,1,i) = 1;
            FBM2(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function X = ConvCM(X,FKM,nl,nc,L)

    if nargin == 3
        [L,n] = size(X);
        nc = n/nl;
    end
    X = conv2mat(real(ifft2(fft2(conv2im(X,nl,nc,L)).*FKM)));
end

function X = conv2mat(X,nl,nc,L)
    if ndims(X) == 3
        [nl,nc,L] = size(X);
        X = reshape(X,nl*nc,L)';
    elseif ndims(squeeze(X)) == 2
        L = 1;
        [nl,nc] = size(X);
        X = reshape(X,nl*nc,L)';
    end
end

function [M, Y] = createSubsampling(Yim,d,nl,nc,L)

    % subsampling matrix
    M = zeros(nl,nc,L);
    indexes = cell([L 1]);

    for i=1:L
        im = ones(floor(nl/d(i)),floor(nc/d(i)));
        maux = zeros(d(i));
        maux(1,1) = 1;
    
        M(:,:,i) = kron(im,maux);
        indexes{i} = find(M(:,:,i) == 1);
        Y(i,indexes{i}) = conv2mat(Yim{i},nl/d(i),nc/d(i),1);
    end
end

function [Yim] = unnormaliseData(Yim, av)
    if iscell(Yim)
        % mean squared power = 1
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



function W = computeWeights(Y,d,sigmas,nl)

    hr_bands = d==1;
    hr_bands = find(hr_bands)';
    for i=hr_bands
        grad(:,:,i) = imgradient(conv2im(Y(i,:),nl),'intermediate').^2;
    end
    grad = sqrt(max(grad,[],3));
    grad = grad / quantile(grad(:),0.95);

    Wim = exp(-grad.^2/2/sigmas^2);
    Wim(Wim<0.5) = 0.5;

    W = conv2mat(Wim,nl);
end

function [Y1,Y2] = regularization(X1,X2,tau,mu,W,q)
    Wr = q*W;
    %Wr=ones(size(Wr));
    Y1 = (mu*X1)./(mu + tau*Wr);
    Y2 = (mu*X2)./(mu + tau*Wr);            
end


function X = conv2im(X,nl,nc,L)

    if size(X,2)==1
        X = conv2mat(X,nl,nc,L);
    end
    if nargin == 2
        [L,n] = size(X);
        if n==1
            X = conv2mat(X,nl,nc,L);
        end
        nc = n/nl;
    end
    X = reshape(X',nl,nc,L);
end






function [J,gradJ,AtAg] = grad_cost_G(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W)
    X=F*Z;
    BX=ConvCM(X,FBM,nl);
    HtHBX=Mask.*BX;
    ZH=ConvCM(Z,FDHC,nl);
    Zv=ConvCM(Z,FDVC,nl);
    ZHW=ZH.*W;
    ZVW=Zv.*W;
    grad_pen=ConvCM(ZHW,FDH,nl)+ConvCM(ZVW,FDV,nl);
    AtAg = F'*ConvCM(HtHBX,conj(FBM),nl)+2*tau*(q*ones(1,nl*nc)).*grad_pen;
    gradJ=AtAg-UBTMTy;
    J = 1/2 * sum( sum( Z .* AtAg ) ) - sum( sum( Z.*UBTMTy ) );     
end

function [ Z ] = CG(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W)
    maxiter = 1000;
    tolgradnorm = 0.1;  %1e-6;   %0.1 
    [cost,grad] = grad_cost_G(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
    gradnorm = norm(grad(:));
    iter = 0;
    res = -grad;
    while ( gradnorm > tolgradnorm & iter < maxiter ) 
        iter = iter + 1;
       % fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);      
        if( iter == 1 )
            desc_dir = res;
        else
            beta = ( res(:).' * res(:) ) / ( old_res(:).' * old_res(:) );
            desc_dir = res + beta * desc_dir;
        end
        [~, ~, AtAp] = grad_cost_G(desc_dir,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
        alpha = ( res(:).' * res(:) ) / ( desc_dir(:).' * AtAp(:) );
        Z1 = Z + alpha * desc_dir;
        old_res = res;
        res = res - alpha* AtAp;
        gradnorm = norm( res(:) );
        % Transfer iterate info
        Z = Z1;
    end
end

%%% Performance Measures

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Spectral Angle Mapper (SAM).
% 
% Interface:
%           [SAM_index,SAM_map] = SAM(I1,I2)
%
% Inputs:
%           I1:         First multispectral image;
%           I2:         Second multispectral image.
% 
% Outputs:
%           SAM_index:  SAM index;
%           SAM_map:    Image of SAM values.
% 
% References:
%           [Yuhas92]   R. H. Yuhas, A. F. H. Goetz, and J. W. Boardman, "Discrimination among semi-arid landscape endmembers using the Spectral Angle Mapper (SAM) algorithm," 
%                       in Proceeding Summaries 3rd Annual JPL Airborne Geoscience Workshop, 1992, pp. 147ï¿½149.
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, ï¿½A Critical Comparison Among Pansharpening Algorithmsï¿½, 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SAM_index,SAM_map] = SAM(I1,I2)

    [M,N,~] = size(I2);
    prod_scal = dot(I1,I2,3); 
    norm_orig = dot(I1,I1,3);
    norm_fusa = dot(I2,I2,3);
    prod_norm = sqrt(norm_orig.*norm_fusa);
    prod_map = prod_norm;
    prod_map(prod_map==0)=eps;
    SAM_map = acos(prod_scal./prod_map);
    prod_scal = reshape(prod_scal,M*N,1);
    prod_norm = reshape(prod_norm, M*N,1);
    z=find(prod_norm==0);
    prod_scal(z)=[];prod_norm(z)=[];
    angolo = sum(sum(acos(prod_scal./prod_norm)))/(size(prod_norm,1));
    SAM_index = real(angolo)*180/pi;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Erreur Relative Globale Adimensionnelle de Synthï¿½se (ERGAS).
% 
% Interface:
%           ERGAS_index = ERGAS(I1,I2,ratio)
%
% Inputs:
%           I1:             First multispectral image;
%           I2:             Second multispectral image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
% 
% Outputs:
%           ERGAS_index:    ERGAS index.
% References:
%           [Ranchin00]     T. Ranchin and L. Wald, ï¿½Fusion of high spatial and spectral resolution images: the ARSIS concept and its implementation,ï¿½
%                           Photogrammetric Engineering and Remote Sensing, vol. 66, no. 1, pp. 49ï¿½61, January 2000.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, ï¿½A Critical Comparison Among Pansharpening Algorithmsï¿½, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ERGAS_index = ERGAS(I1,I2,ratio)

    I1 = double(I1);
    I2 = double(I2);

    Err=I1-I2;
    ERGAS_index=0;
    for iLR=1:size(Err,3),
        ERGAS_index = ERGAS_index+mean2(Err(:,:,iLR).^2)/(mean2((I1(:,:,iLR))))^2;   
    end

    ERGAS_index = (100/ratio) * sqrt((1/size(Err,3)) * ERGAS_index);

end

