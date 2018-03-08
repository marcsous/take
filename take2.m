function [ksp mask] = take2(data,mask,varargin)
% [ksp mask] = take2(data,mask,varargin)
%
% Trimmed autocalibrating k-space estimation in 2d based on
% structured low rank matrix completion.
%
% Inputs:
%  -data [nx ny nc]: 2d kspace data array from nc coils
%  -mask [nx ny]: 2d sampling mask (or 1d vector, legacy) 
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nc]: 2d kspace data array from nc coils
%  -mask [nx ny]: 2d sampling mask with outliers trimmed
%
% References:
%  -Bydder M et al. TAKE. Magnetic Resonance Imag 2017;43:88
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%
%% setup

% default options
opts.width = 5; % kernel width
opts.radial = 0; % use radial kernel
opts.tol = 5e-4; % relative tolerance
opts.noise = []; % noise std, if available
opts.nstd = 4; % no. stds considered outlier
opts.proj = 2; % projection dimension (0, 1 or 2)
opts.loraks = 0; % use phase constraint (loraks)
opts.center = []; % center of kspace, if available

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
if ndims(data)<2 || ndims(data)>3
    error('Argument ''data'' must be a 3d array.')
end
[nx ny nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,3); % 2d mask [nx ny]
    warning('Argument ''mask'' not supplied - guessing.')
elseif isvector(mask)
    if ~isa(mask,'logical')
        index = mask; % old way, legacy
        mask = false(1,ny); mask(index) = 1;
    end
    mask = repmat(reshape(mask,1,ny),nx,1);
end
mask = reshape(mask,nx,ny); % catch size mismatch

% convolution kernel indicies
[x y] = ndgrid(-ceil(opts.width/2):ceil(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);

% dimensions of the matrix
opts.dims = [nx ny nc nk];
if opts.loraks; opts.dims = [opts.dims 2]; end

% estimate center of kspace
if isempty(opts.center)
    [~,k] = max(reshape(data,[],nc));
    [x y] = ind2sub([nx ny],k);
    opts.center(1) = round(median(x));
    opts.center(2) = round(median(y));
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% memory required for calibration matrix
k = gather(data(1)) * 0; % single or double
bytes = 2 * prod(opts.dims) * getfield(whos('k'),'bytes');

% density of calibration matrix
density = nnz(mask) / numel(mask);

% display
t = tic;
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Sampling density = %f\n',density);
fprintf('Matrix = %ix%i (%.1f Mb)\n',nx*ny,prod(opts.dims(3:end)),bytes/1e6);

%% see if gpu is possible

try
    gpu = gpuDevice;
    if gpu.AvailableMemory < 4*bytes; error('GPU memory too small.'); end
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    data = gpuArray(data);
    mask = gpuArray(mask);
catch ME
    data = gather(data);
    mask = gather(mask);
    warning([ME.message ' Using CPU.'])
end

%% POCS iterations - solve for ksp

ksp = zeros(nx,ny,nc,'like',data);

for iter = 1:1000000

    % data consistency
    ksp = bsxfun(@times,data,mask) + bsxfun(@times,ksp,~mask);

    % make calibration matrix
    A = make_data_matrix(ksp,opts);

    % row space, singular values
    [V S] = svd(A'*A);
    S = sqrt(diag(S));

    % initialize noise floor
    if isempty(opts.noise)
        
        % find a small singular value
        for j = 1:numel(S)
            h = hist(S(j:end));
            [~,k] = max(h);
            if k>1; break; end
        end
        noise_floor = median(S(j:end));
        if noise_floor==0; error('noise floor estimation failed.'); end

        % estimate noise std
        opts.noise = noise_floor / sqrt(2*density*size(A,1));
        disp(['Estimated noise std = ' num2str(opts.noise)]);

    else
        noise_floor = opts.noise * sqrt(2*density*size(A,1));
    end

    % minimum variance filter
    f = max(0,1-noise_floor.^2./S.^2);
    A = A * (V * diag(f) * V');
    
    % copies from rank reduction
    copy = undo_data_matrix(A,opts);
    
    % irls+huber (aka geometric median)
    for j = 1:5
        w = abs(bsxfun(@minus,copy,ksp));
        w = 1./max(w,opts.noise*1e-7);
        ksp = sum(w.*copy,4)./sum(w,4);        
    end

    % convergence
    normA(iter) = sum(S);
    if iter==1; old = NaN; end
    tol(iter) = norm(ksp(:)-old(:))/norm(ksp(:));
    converged = tol(iter) < opts.tol;
    old = ksp;

    % residual (abs is less sensitive to translations)
    r = abs(data) - abs(ksp);
    %r = data - ksp;

    % project over coil dimension
    r = sum(abs(r),3);
    
    % project over kspace dimension
    if opts.proj
        r(~mask) = 0;
        r = sum(r,opts.proj);
        index = find(any(mask,opts.proj));
    else    
        index = find(mask);
    end
    r = r(index);
    
    % median and std dev of residual
    med_r = median(r);
    std_r = median(abs(r - med_r)) * 1.4826;

    % trim the worst data
    if converged

        % no. stds away from the median
        [nstd k] = max(r/std_r-med_r/std_r);

        if nstd < opts.nstd
            rejected = 0; % reject nothing
        else
            rejected = index(k);
            if opts.proj==0; mask(rejected) = 0; end % reject worst point
            if opts.proj==1; mask(:,rejected) = 0; end % reject worst col
            if opts.proj==2; mask(rejected,:) = 0; end % reject worst row
        end

        % display info
        if ~exist('ntrim','var')
            ntrim = 0;
            disp('-----------------------------------------------------')
            disp(' Count   ||A||  Iteration Trimmed  nstd   std    Time');
        end
        ntrim = ntrim+1; % no. trimmed points/cols/rows
        fprintf('%5i %9.2e %6i',ntrim,normA(iter),iter);
        fprintf('%9i %6.1f %9.2e %4.0f\n',rejected,nstd,std_r,toc(t));

    end
    
    % display plots every 10 iterations
    if mod(iter,10)==1 || converged

        if iter<50
            % plot singular values
            subplot(1,3,1)
            plot(S/S(1));
            hold on; plot(max(f,min(ylim)),'--'); hold off
            line(xlim,gather([1 1]*noise_floor/S(1)),'linestyle',':','color','black');
            legend({'singular vals.','min. var. filter','noise floor'});
            title(sprintf('rank %i',nnz(f))); xlim([0 numel(S)+1]);
        else
            % show residual norm
            subplot(1,3,1);
            if opts.proj
                plot(index,r/std_r); title('residuals'); ylabel('||r||_1  /  std dev');
                if opts.proj==1; xlim([0 ny+1]); xlabel('dim 2'); else xlim([0 nx+1]); xlabel('dim 1'); end
                line(xlim,[0 0]+med_r/std_r+opts.nstd,'linestyle','--','color','red');
                line(xlim,[0 0]+med_r/std_r-opts.nstd,'linestyle','--','color','red');
                line(xlim,[0 0]+med_r/std_r,'color','red'); axis tight;
            else
                temp = zeros(nx,ny,'like',r); temp(index) = r/std_r; ims(temp);
                xlabel('dim 2'); ylabel('dim 1'); title('residual map'); colorbar
            end
        end
        % show current image
        subplot(1,3,2);
        ims(sum(abs(ifft2(ksp)),3));
        xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));
        % plot change in norm and tol
        subplot(1,4,4);
        [h,~,~] = plotyy(1:iter,max(tol,opts.tol),1:iter,normA);
        set(h(1),'YScale','log'); set(h(2),'YScale','log');
        title('metrics'); legend({'||Δk||/||k||','||A||_* norm'});
        xlim(h(1),[0 iter+1]); xlim(h(2),[0 iter+1]); xlabel('iters');
        drawnow;
 
    end

    % finish when nothing left to do
    if converged && rejected==0; break; end

end

% return on CPU
ksp = gather(ksp);
mask = gather(mask);

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
    B = A(opts.flip.x,opts.flip.y,:,:);
    A = cat(4,A,conj(B));
end

A = reshape(A,nx*ny,[]);

%% undo calibration matrix
function A = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,[]);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(A(:,:,:,k),-[x y]);

    if opts.loraks
        B = A(opts.flip.x,opts.flip.y,:,k+nk);
        A(:,:,:,k+nk) = circshift(conj(B),-[x y]);
    end
end
