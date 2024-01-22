function [ksp mask flag] = take2(data,varargin)
% [ksp mask flag] = take2(data,varargin)
%
% Trimmed autocalibrating k-space estimation in 2D
% based on structured low rank matrix completion.
% Uses an heuristic approach to remove outliers.
%
% Performance depends on the noise std which should
% ideally be provided. Other key parameters are
% tolerance (smaller is better/slower) and no. stds
% to define the outlier threshold.
%
% Inputs:
%  -data [nx ny nc]: kspace data from nc coils
%  -varargin: option/value pairs (e.g. 'nstd',4)
%
% Outputs:
%  -ksp [nx ny nc]: kspace data from nc coils
%  -flag is zero for successful termination
%
% References:
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Bydder M et al. TAKE. Magnetic Resonance Imag 2017;43:88

%% setup

% default options
opts.width = 4; % kernel width (default 4)
opts.radial = 1; % use radial kernel (0 or 1)
opts.loraks = 1; % use phase constraint (0 or 1)
opts.tol = 1e-5; % relative tolerance (1e-5)
opts.maxit = 1e4; % maximum no. iterations (1e4)
opts.p = 2; % singular filter shape (>=1) 
opts.irls = 3; % no. irls iterations (0=mean)
opts.nstd = 4; % outlier threshold (no. std devs)
opts.readout = 2; % readout dimension (0, 1 or 2)
opts.std = []; % noise std dev, if available
opts.power = 0.5; % density weighting power (0=off)
opts.errors = []; % known errors (for validation)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks
if ndims(data)<2 || ndims(data)>3 || ~isfloat(data) || isreal(data)
    error('Argument ''data'' must be a 3d complex float array.')
end
[nx ny nc] = size(data);

% readout dimension
if ~ismember(opts.readout,[0 1 2])
    error('readout can only be 0, 1 or 2.');
end

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = hypot(x,y)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.mask = k;

% estimate center of kspace
[~,k] = max(reshape(data,[],nc));
[x y] = ind2sub([nx ny],k);
opts.center(1) = round(gather(median(x)));
opts.center(2) = round(gather(median(y)));

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% dimensions of the matrix
opts.dims = [nx ny nc nk 1+opts.loraks];

% sampling mask (same for all coils)
mask = any(data,3);

% density
matrix_density = nnz(mask) / numel(mask);
sample_density = calc_sample_density(mask,opts);

% estimate noise std
std_estimated = isempty(opts.std);
if std_estimated; opts.std = estimate_std(data,mask); end
noise_floor = opts.std * sqrt(nnz(mask));

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Matrix density = %f\n',matrix_density);

%% see if gpu is possible
try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    mask = gpuArray(mask);
    data = gpuArray(data);
    opts.flip.x = gpuArray(opts.flip.x);
    opts.flip.y = gpuArray(opts.flip.y);
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    mask = gather(mask);
    data = gather(data);
    opts.flip.x = gather(opts.flip.x);
    opts.flip.y = gather(opts.flip.y);
    warning('%s Using CPU.', ME.message);
end

%% Cadzow algorithm - solve for ksp

ksp = data; 
   
count = 0; % total number of rejections
restart = 0; % iteration no. of last rejection
t(1:2) = tic; % timers (1)=global (2)=display

for iter = 1:opts.maxit
    
    % data consistency
    ksp = ksp + bsxfun(@times,data-ksp,mask);
    
    % make calibration matrix
    [A opts] = make_data_matrix(ksp,opts);

    % row space, singular values (A'A=V'S'*S*V)
    [V S] = svd(A'*A);
    S = sqrt(diag(S));
    
    % singular value filter
    f = max(0,1-noise_floor.^opts.p./S.^opts.p); 
    A = A * (V * diag(f) * V');

    % undo hankel structure
    [A opts] = undo_data_matrix(A,opts);

    % combine redundant copies (mean or irls)
    ksp = mean(A,4);
    for j = 1:opts.irls
        w = abs(bsxfun(@minus,A,ksp));
        w = 1./hypot(w,opts.std*1e-2); % "small" tuning parameter
        ksp = sum(w.*A,4) ./ sum(w,4);
    end

    % convergence metrics
    snorm(iter) = norm(S,opts.p);
    if iter-restart<10 || snorm(iter)<snorm(iter-1)
        tol = NaN;
    else
        tol = (snorm(iter)-snorm(iter-1)) / snorm(iter);
    end

    % residual: abs(data-ksp) is too(?) sensitive to phase errors
    r = abs(abs(ksp)-abs(data));

    % project over coil dimension
    r = sum(r,3) .* mask;

    % heuristic to inhibit rejection of neigbors
    r = r .* power(sample_density,opts.power);

    % project over readout dimension
    if opts.readout; r = sum(r,opts.readout); end

    % center and normalize by (robust) std
    r = reshape(r,[],1);
    [k,~,v] = find(r);
    r(k) = r(k) - median(v);
    rstd = 1.4826 * median(abs(r(k)));
    r = r / rstd;

    % trim after convergence 
    if tol<opts.tol
        
        % find the worst data
        [nstd reject] = max(r);
        
        if nstd < opts.nstd
            reject = 0; % trim nothing
        else
            if opts.readout==2; x = reject; y = ':'; end % row
            if opts.readout==1; y = reject; x = ':'; end % col
            if opts.readout==0; [x y] = ind2sub([nx ny],reject); end % point
            
            % update parameters
            mask(x,y) = 0; r(reject) = 0;
            restart = iter; count = count+1;             
            matrix_density = nnz(mask) / numel(mask);
            sample_density = calc_sample_density(mask,opts);
            if std_estimated; opts.std = estimate_std(data,mask); end
            noise_floor = opts.std * sqrt(nnz(mask));
            
            % display progress
            if count==1
                fprintf('Iterations per second: %.2f\n',(iter-1) / toc(t(1)));
                disp('-----------------------------------------------------');
                disp('Count  ||A||   Iter  Trimmed   rstd     noise    Time');
                disp('-----------------------------------------------------');
            end
            fprintf('%3i %9.2e %5i',count,snorm(iter),iter);
            if isempty(opts.errors)
                fprintf(' %6i  ',reject);
            else
                fprintf('%6i(%1i)',reject,ismember(reject,opts.errors));
            end
            fprintf('%9.2e %9.2e %5.0f\n',rstd,noise_floor,toc(t(1)));
        end
        
    elseif iter==1 || toc(t(2)) > 1 % update every second
        
        display(S,f,r,ksp,data,iter,snorm,tol,mask,opts); t(2) = tic;
        
    end
    
    % finish
    if tol<opts.tol && reject==0; break; end

end

% return on CPU
ksp = gather(ksp);
mask = gather(mask);
flag = reject;

%% sample density in kspace (approximate)
function d = calc_sample_density(mask,opts);
kernel = fftn(opts.kernel.mask,opts.dims(1:2));
d = ifftn(bsxfun(@times,fftn(mask),kernel),'symmetric');
d = circshift(d,-floor(size(opts.kernel.mask)/2));
d = max(d.*mask,0)/nnz(opts.kernel.mask);

%% estimate noise std from data (heuristic)
function noise_std = estimate_std(data,mask)
tmp = bsxfun(@times,data,mask); tmp = nonzeros(tmp);
tmp = sort([real(tmp); imag(tmp)]); % separate real/imag for median
k = ceil(numel(tmp)/5); tmp = tmp(k:end-k+1); % trim 20% off both ends
noise_std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2); % robust std

%% make calibration matrix
function [A opts] = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
nk = opts.dims(4);

% precompute the circshifts with fast indexing
if ~isfield(opts,'ix')
    opts.ix = repmat(1:uint32(nx*ny*nc),[1 nk]);
    opts.ix = reshape(opts.ix,[nx ny nc nk]);
    for k = 1:nk
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        opts.ix(:,:,:,k) = circshift(opts.ix(:,:,:,k),[x y]);
    end
    if isa(data,'gpuArray'); opts.ix = gpuArray(opts.ix); end
end
A = data(opts.ix);

if opts.loraks
    A = cat(5,A,conj(A(opts.flip.x,opts.flip.y,:,:)));
end

A = reshape(A,nx*ny,[]);

%% undo calibration matrix
function [A opts] = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,nk,[]);

if opts.loraks
    A(opts.flip.x,opts.flip.y,:,:,2) = conj(A(:,:,:,:,2));
end

% precompute the circshifts with fast indexing
if ~isfield(opts,'xi')
    opts.xi = reshape(1:uint32(numel(A)),size(A));
    for k = 1:nk
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        opts.xi(:,:,:,k,:) = circshift(opts.xi(:,:,:,k,:),-[x y]);
    end
    if isa(A,'gpuArray'); opts.xi = gpuArray(opts.xi); end
end
A = A(opts.xi);

A = reshape(A,nx,ny,nc,[]);

%% show plots of various things (slow)
function display(S,f,r,ksp,data,iter,snorm,tol,mask,opts)
[nx ny nc] = size(ksp);
% prefer ims over imagesc
if exist('ims','file'); imagesc = @(x)ims(x,-0.99); end
% singular values
subplot(2,4,[1 5]); plot(S/S(1)); hold on; plot(f,'--'); hold off
xlim([0 numel(f)]); title(sprintf('rank %i/%i',nnz(f),numel(f)));
line(xlim,[0 0]+gather(S(nnz(f))/S(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});
% residual norm plot
subplot(2,4,2); [k,~,v] = find(r); plot(k,v); ylabel('r / std');
if opts.readout==0; xlabel('dims'); end
if opts.readout==1; xlabel('dim 2'); end
if opts.readout==2; xlabel('dim 1'); end
axis tight; line(xlim,[0 0]+opts.nstd,'linestyle','--','color','red');
line(xlim,[0 0],'linestyle','-','color','red'); title('residual plot');
% residual norm map
subplot(2,4,6); tmp = sum(abs(data-ksp),3).*mask;
if opts.readout==2; tmp = tmp'; end; imagesc(log(tmp)); title('residual map');
if opts.readout<=1; xlabel('dim 2'); ylabel('dim 1'); end
if opts.readout==2; xlabel('dim 1'); ylabel('dim 2'); end
% current image
subplot(2,4,[3 7]); tmp = sum(abs(ifft2(ksp)),3);
if max(tmp(:,1))>max(tmp(:,floor(ny/2)+1)); tmp = fftshift(tmp); end
imagesc(tmp); xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));
% change in norm
subplot(2,4,[4 8]); semilogy(1:iter,snorm); xlim([0 iter]); xlabel('iters');
title(sprintf('tol %.2e',tol)); grid on; legend('||A||_p','location','northwest');
drawnow;