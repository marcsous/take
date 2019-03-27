function [ksp mask flag] = take2(data,mask,varargin)
% [ksp mask flag] = take2(data,mask,varargin)
%
% Trimmed autocalibrating k-space estimation in 2D
% based on structured low rank matrix completion.
% Uses a heuristic approaches to remove outliers.
%
% Singular value filtering requires noise estimate
% which is a key parameter. The other key parameter
% is tolerance (smaller is better but slower).
%
% Inputs:
%  -data [nx ny nc]: kspace data from nc coils
%  -mask: 1D/2D/3D sampling mask (or 1D indices)
%  -varargin: option/value pairs (e.g. 'nmad',4)
%
% Outputs:
%  -ksp [nx ny nc]: kspace data from nc coils
%  -mask [nx ny nc]: updated sampling mask
%  -flag is zero for successful termination
%
% References:
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Bydder M et al. TAKE. Magnetic Resonance Imag 2017;43:88

%% setup

% default options
opts.width = 5; % kernel width (default 5)
opts.radial = 0; % use radial kernel (0 or 1)
opts.loraks = 0; % use phase constraint (0 or 1)
opts.tol = 5e-4; % relative tolerance (5e-4)
opts.maxit = 1e4; % maximum no. iterations (1e4)
opts.minit = 1e1; % minimum no. iterations (1e1)
opts.irls = 5; % no. irls iterations (0=use mean)
opts.nmad = 5; % outlier threshold (mad=1.4826*std)
opts.proj = 2; % projection dimension (0, 1 or 2)
opts.noise = []; % noise std, if available
opts.center = []; % center of kspace, if available
opts.errors = []; % known errors (for validation)
opts.display = 20; % no. iterations between plots
opts.power = 0.5; % density weighting power (0=off)

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
opts.dims = [nx ny nc nk 1+opts.loraks];

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

%% Cadzow algorithm - solve for ksp

t = tic;
old = NaN;
converged = 0;
noise_floor = -1;

ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit
    
    % data consistency
    ksp = bsxfun(@times,data,mask)+bsxfun(@times,ksp,~mask);
    
    % make calibration matrix
    A = make_data_matrix(ksp,opts);
    
    % row space, singular values (A'A=U*W*V')
    [V W] = svd(A'*A);
    W = sqrt(diag(W));
    
    % estimate noise floor of singular values (heuristic)
    if ~isempty(opts.noise)
        noise_floor = opts.noise * sqrt(matrix_density*nx*ny);
    elseif noise_floor==-1
        for j = 1:numel(W)
            h = hist(W(j:end));
            [~,k] = max(h);
            if k>1; break; end
        end
        noise_floor = median(W(j:end)); % center of Marcenko-Pastur distribution?
    end
    noise_std = noise_floor / sqrt(matrix_density*nx*ny); % noise std estimate

    % minimum variance filter (De Moor B. IEEE Trans Sig Proc 1993;41:2826)
    f = max(0,1-noise_floor.^2./W.^2); 
    F = V * diag(f) * V';
    A = A * F;

    % undo Hankel stucture
    A = undo_data_matrix(A,opts);
    
    % A should be rank nc, combine redundant copies
    A = reshape(A,nx,ny,nc,[]);
    if opts.irls==0
        ksp = mean(A,4);
    else
        for j = 1:opts.irls
            w = abs(bsxfun(@minus,A,ksp));
            w = 1./max(w,noise_std*1e-7);
            ksp = sum(w.*A,4)./sum(w,4);
        end
    end

    % check convergence: min(normA) not adequate so use tol
    normA(iter) = sum(W);
    tol(iter) = norm(ksp(:)-old(:)) / norm(ksp(:));
    converged = converged + (tol(iter) < opts.tol);
    old = ksp;
    
    % residual... or possibly abs(abs(data)-abs(ksp))?
    r = abs(data-ksp);
 
    % project over coil dimension
    r = sum(r,3);

    % heuristic to inhibit rejection of too many neigboring lines
    r = (r-median(r(mask))) .* power(sample_density,opts.power);

    % project over kspace dimension
    if opts.proj
        r = sum(r,opts.proj);
    end

    % center and normalize
    r = reshape(r,[],1);
    [k,~,v] = find(r);
    r(k) = r(k) - median(v);
    mad = median(abs(r(k)));
    r = r / mad;

    % find the worst data
    [nmad reject] = max(r);

    % trim the worst data
    if converged==opts.minit
        
        if nmad < opts.nmad
            reject = 0; % trim nothing
        else
            if opts.proj==0; [x y] = ind2sub([nx ny],reject); end % point
            if opts.proj==1; y = reject; x = ':'; end % col
            if opts.proj==2; x = reject; y = ':'; end % row
            mask(x,y) = 0; % trim point/col/row
            
            % catastrophic failure (center of kspace gone!)
            idx = mod(opts.center(1)+opts.kernel.x-1,nx)+1;
            idy = mod(opts.center(2)+opts.kernel.y-1,ny)+1;
            if nnz(mask(idx,idy))==0; reject = -1; end
        end
        
        % display trim info
        if ~exist('ntrim','var')
            ntrim = 1; % counter for no. trimmed points/cols/rows
            disp('----------------------------------------------------------');
            disp('Count  ||A||   Iter  Trimmed nmad    mad      Noise   Time');
            disp('----------------------------------------------------------');
        end
        fprintf('%3i %9.2e %5i',ntrim,normA(iter),iter);
        if isempty(opts.errors)
            fprintf(' %6i  ',reject);
        else
            fprintf('%6i(%1i)',reject,ismember(reject,opts.errors));
        end
        fprintf('%5.1f %9.2e %9.2e %4.0f\n',min(nmad,99.9),mad,noise_std,toc(t));

        % update parameters if we trimmed something
        if reject>0
            ntrim = ntrim+1; % increment counter
            matrix_density = nnz(mask) / numel(mask);
            sample_density = calc_sample_density(mask,opts);
            converged = 0; % reset minit counter            
            noise_floor = -1; % trigger recalculation
        end
        
    end
    
    % display plots
    if mod(iter,opts.display)==1 || converged==opts.minit
        % singular values
        subplot(2,4,[1 5]); plot(W/W(1)); hold on; plot(f,'--'); hold off
        xlim([0 numel(f)+1]); title(sprintf('rank %i/%i',nnz(f),numel(f)));
        line(xlim,[0 0]+gather(W(nnz(f))/W(1)),'linestyle',':','color','black');
        legend({'singular vals.','sing. val. filter','noise floor'});
        % residual norm plot
        subplot(2,4,2); [k,~,v] = find(r); plot(k,v); ylabel('r / mad');
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
        xlim(g,[0 iter+1]); xlabel('iters');
        title('metrics'); legend({'||Δk||_F','||A||_*'}); drawnow;
    end
    
    % finish when nothing left to do
    if converged==opts.minit && reject<=0
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
flag = reject;

%% approximate sample density in kspace
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
