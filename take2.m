function [ksp mask flag] = take2(data,mask,varargin)
% [ksp mask flag] = take2(data,mask,varargin)
%
% Trimmed autocalibrating k-space estimation in 2D
% based on structured low rank matrix completion.
%
% Singular value filtering is done based on opts.noise
% which is a key parameter that affects image quality.
%
% Inputs:
%  -data [nx ny nc]: 2D kspace data from nc coils
%  -mask: 1D/2D/3D sampling mask (or 1D indices)
%  -varargin: option/value pairs (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nc]: 2D kspace data from nc coils
%  -mask [nx ny nc]: sampling mask with outliers trimmed
%  -flag is 0 for successful termination
%
% References:
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%  -Bydder M et al. TAKE. Magnetic Resonance Imag 2017;43:88

%% setup

% default options
opts.width = 6; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % use phase constraint
opts.tol = 1e-3; % relative tolerance
opts.maxit = 10000; % maximum no. iterations
opts.minit = 5; % min. no. iterations after trim
opts.irls = 5; % no. irls iterations (0=use mean)
opts.nmad = 3; % outlier threshold (mad=1.4826*std)
opts.proj = 2; % projection dimension [0 1 2]
opts.noise = []; % noise std, if available
opts.center = []; % center of kspace, if available
opts.errors = []; % known errors (for validation)
opts.display = 10; % no. iterations to update display

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        warning('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks - try to be flexible
if ndims(data)<2 || ndims(data)>3
    error('Argument ''data'' must be a 3d array.')
end
[nx ny nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,3); % 2d mask [nx ny]
    warning('''mask'' not supplied - guessing.')
elseif isvector(mask)
    if any(mask~=0 & mask~=1) 
        if any(mask<1 | mask>ny | mod(mask,1))
            error('''mask'' is incompatible.');
        end
        index = mask; % linear indices
        if opts.proj<=1; mask = false(1,ny); end
        if opts.proj==2; mask = false(nx,1); end
        mask(index) = 1; % binary mask
    end
    if opts.proj<=1; mask = repmat(reshape(mask,1,ny),[nx 1]); end
    if opts.proj==2; mask = repmat(reshape(mask,nx,1),[1 ny]); end
end
if nnz(mask~=0 & mask~=1); error('''mask'' is incompatible.'); end
mask = reshape(mask==1,[nx ny]); % ensure size/class compatibility

% projection dimension
if ~isscalar(opts.proj) || opts.proj<0 || opts.proj>2 | mod(opts.proj,1)
    error('projection can only be 0, 1 or 2.');
end

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.mask = k;

% dimensions of the matrix
opts.dims = [nx ny nc nk 1];
if opts.loraks; opts.dims(5) = 2; end

% estimate center of kspace
if isempty(opts.center)
    [~,k] = max(reshape(data,[],nc));
    [x y] = ind2sub([nx ny],k);
    opts.center(1) = round(gather(median(x)));
    opts.center(2) = round(gather(median(y)));
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% density
matrix_density = nnz(mask) / numel(mask);
sample_density = calc_sample_density(mask,opts);

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Matrix density = %f\n',matrix_density);

%% see if gpu is possible
try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    mask = gather(mask);
    warning('%s Using CPU.', ME.message);
end

%% Cadzow algorithm

ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit
    
    % data consistency
    ksp = bsxfun(@times,data,mask)+bsxfun(@times,ksp,~mask);
    
    % make calibration matrix
    A = make_data_matrix(ksp,opts);
    
    % row space, singular values
    [V W] = svd(A'*A);
    W = sqrt(diag(W));
    
    % estimate noise floor from singular values (ad hoc)
    if ~isempty(opts.noise)
        sigma = opts.noise * sqrt(matrix_density*nx*ny);
    elseif ~exist('sigma','var')
        for j = 1:numel(W)
            h = hist(W(j:end));
            [~,k] = max(h);
            if k>1; break; end
        end
        sigma = median(W(j:end)); % center of Marcenko-Pastur distribution?
    end
    noise_est = sigma / sqrt(matrix_density*nx*ny); % noise estimated from sigma

    % minimum variance filtering
    f = max(0,1-sigma.^2./W.^2); 
    F = V * diag(f) * V';
    A = A * F;

    % undo Hankel stucture
    A = undo_data_matrix(A,opts);
    
    % A should be rank nc - combine redundant copies
    A = reshape(A,nx,ny,nc,[]);

    if opts.irls==0
        ksp = mean(A,4);
    else
        for j = 1:opts.irls
            w = abs(bsxfun(@minus,A,ksp));
            w = 1./max(w,noise_est*1e-7);
            ksp = sum(w.*A,4)./sum(w,4);
        end
    end

    % convergence (increment until it reaches opts.minit)
    if iter==1
        t = tic;
        converged = 0;
        tol(iter) = opts.tol;
    else
        tmp = norm(ksp(:)-old(:)) / norm(ksp(:));
        tol(iter) = gather(tmp);
        converged = converged + (tol(iter) < opts.tol);
    end
    old = ksp;
    normA(iter) = sum(W);
    
    % residual
    r = abs(data-ksp);
 
    % project over coil dimension and center
    r = median(r,3);
    r = r - median(r(mask));

    % scale by sample density (to avoid trimming consecutive lines)
    r = r .* sample_density;
    
    % project over kspace dimension, center and normalize
    r(~mask) = NaN;
    if opts.proj
        r = median(r,opts.proj,'omitnan');
        r = r - median(r,'omitnan');
        mad = median(abs(r),'omitnan');
    else
        r = r - median(r(:),'omitnan');
        mad = median(abs(r(:)),'omitnan');
    end
    r = r / mad;
    
    % find the worst data
    [nmad flag] = max(r(:));
    
    % trim the worst data
    if converged>=opts.minit
        
        if nmad < opts.nmad
            flag = 0; % trim nothing
        else
            if opts.proj==0; [x y] = ind2sub([nx ny],flag); end % point
            if opts.proj==1; y = flag; x = ':'; end % col
            if opts.proj==2; x = flag; y = ':'; end % row
            mask(x,y,:) = 0; % trim point/col/row
            
            % catastrophic failure (center of kspace gone!)
            idx = mod(opts.center(1)+opts.kernel.x-1,nx)+1;
            idy = mod(opts.center(2)+opts.kernel.y-1,ny)+1;
            if nnz(mask(idx,idy))==0; flag = -1; end
        end
        
        % display trim info
        if ~exist('ntrim','var')
            ntrim = 0; % no. trimmed points/cols/rows
            disp('-------------------------------------------------------------------');
            disp('Count  ||A||  Tolerance  Iter Trimmed nmad    mad      Noise   Time');
            disp('-------------------------------------------------------------------');
        end
        ntrim = ntrim+1;
        fprintf('%3i %9.2e %8.1e %5i',ntrim,normA(iter),tol(iter),iter);
        if isempty(opts.errors)
            fprintf(' %6i  ',flag);
        else
            fprintf('%6i(%1i)',flag,ismember(flag,opts.errors));
        end
        fprintf('%5.1f %9.2e %9.2e %4.0f\n',min(nmad,99.9),mad,noise_est,toc(t));

        % update parameters if we trimmed something
        if flag>0
            matrix_density = nnz(mask) / numel(mask);
            sample_density = calc_sample_density(mask,opts);
            clear sigma; % trigger recalculation
            converged = 0; % reset minit counter
        end
        
    end
    
    % display plots
    if mod(iter,opts.display)==1 || converged>=opts.minit
        % singular values
        subplot(2,4,[1 5]); plot(W/W(1)); hold on; plot(f,'--'); hold off
        xlim([0 numel(f)+1]); title(sprintf('rank %i/%i',nnz(f),numel(f)));
        line(xlim,[0 0]+gather(W(nnz(f))/W(1)),'linestyle',':','color','black');
        legend({'singular vals.','sing. val. filter','noise floor'});
        % residual norm plot
        subplot(2,4,2); k = find(~isnan(r(:))); plot(k,r(k)); ylabel('r / mad');
        if opts.proj==0; xlim([0 nx*ny+1]); xlabel('dims'); end
        if opts.proj==1; xlim([0 ny+1]); xlabel('dim 2'); end
        if opts.proj==2; xlim([0 nx+1]); xlabel('dim 1'); end
        line(xlim,[0 0]+opts.nmad,'linestyle','--','color','red'); axis tight;
        line(xlim,[0 0],'linestyle','-','color','red'); title('residual plot');
        % residual norm map
        subplot(2,4,6); tmp = bsxfun(@times,data-ksp,mask); tmp = sum(abs(tmp),3);
        if opts.proj==2; tmp = tmp'; end; imagesc(log(tmp)); title('residual map');
        if opts.proj<=1; xlabel('dim 2'); ylabel('dim 1'); end
        if opts.proj==2; xlabel('dim 1'); ylabel('dim 2'); end
        % current image
        subplot(2,4,[3 7]); tmp = sum(abs(ifft2(ksp)),3);
        if tmp(1,1)>tmp(fix(nx/2),fix(ny/2)); tmp = fftshift(tmp); end;
        imagesc(tmp); xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));
        % change in norm and tol
        subplot(2,4,[4 8]); g = plotyy(1:iter,tol,1:iter,normA,'semilogy','semilogy');
        title('metrics'); axis(g,'tight'); xlim(g,[0 iter+1]);
        legend({'||Δk||/||k||','||A||_* norm'}); xlabel('iters'); drawnow;
    end
    
    % finish when nothing left to do
    if converged>=opts.minit && flag<=0
        % show original image
        subplot(2,4,[4 8]); tmp = sum(abs(ifft2(data)),3);
        if tmp(1,1)>tmp(fix(nx/2),fix(ny/2)); tmp = fftshift(tmp); end
        imagesc(tmp); xlabel('dim 2'); ylabel('dim 1'); title('iter 1');
        break;
    end
    
end

% return on CPU
ksp = gather(ksp);
mask = gather(mask);

%% sample density in kspace (approximate)
function d = calc_sample_density(mask,opts);
kernel = fftn(opts.kernel.mask,opts.dims(1:2));
d = ifftn(bsxfun(@times,fftn(mask),kernel),'symmetric');
d = circshift(d,-fix(size(opts.kernel.mask)/2));
d = max(d.*mask,0)/nnz(opts.kernel.mask);

%% make calibration matrix
function A = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
nk = opts.dims(4);

A = zeros(nx,ny,nc,nk,'like',data);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(data,[x y]);
end

if opts.loraks
    A = cat(5,A,conj(A(opts.flip.x,opts.flip.y,:,:)));
end

A = reshape(A,nx*ny,[]);

%% undo calibration matrix
function A = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,nk,[]);

if opts.loraks
    A(opts.flip.x,opts.flip.y,:,:,2) = conj(A(:,:,:,:,2));
end

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k,:) = circshift(A(:,:,:,k,:),-[x y]);
end

A = reshape(A,nx*ny,[]);
