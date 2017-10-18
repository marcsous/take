function [ksp keep] = take2(data,mask,varargin)
% [ksp keep] = take2(data,mask,varargin)
%
% Trimmed autocalibrating k-space estimation based
% on structured matrix completion for 2d datasets.
%
% Inputs:
%  -data [nx ny nc]: 2d kspace data from nc coils
%  -mask [nx ny]: sampling mask (or 1d vector, legacy) 
%  -varargin: option/value pairs (e.g. 'width',7)
%
% Outputs:
%  -ksp [nx ny nc]: 2d kspace data from nc coils
%  -keep [nx ny]: mask of retained kspace samples
%
% References:
%  -Bydder M. Magnetic Resonance Imaging 43:88 (2017)
%  -Haldar JP. IEEE Trans Med Imaging 33:668 (2014)
%  -Shin PJ. Magnetic Resonance Medicine 72:959 (2014)
%
% Notes:
% The gpu option needs MATLAB R2014b (8.4) or later.
%
%% setup

% default options
opts.width = 5; % kernel width
opts.radial = 0; % use radial kernel
opts.tol = 5e-4; % relative tolerance
opts.maxit = 1e6; % max. no. iterations
opts.noise = []; % noise std, if available
opts.nstd = 5; % no. stds to consider outlier
opts.loraks = 0; % smooth phase constraint (loraks)
opts.proj = 2; % projection dimension (0, 1 or 2) 
opts.irls_scale = 5e-4; % irls scale parameter
opts.irls_iters = 5; % no. irls iterations
opts.gpu = gpuDeviceCount; % use gpu, if available
opts.class = 'single'; % data precision (single/double)

% varargin handling - must be in option/value pairs
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be supplied in option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks
if ~exist('data','var') || ndims(data)<2 || ndims(data)>3
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

% center of kspace
if ~isfield(opts,'center')
    [~,k] = max(reshape(data,[],nc));
    [x y] = ind2sub([nx ny],k);
    opts.center(1) = round(mean(x));
    opts.center(2) = round(mean(y));
end

% convolution kernel indicies, centered for loraks
[x y] = ndgrid(-ceil(opts.width/2):ceil(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k) + nx/2 - opts.center(1);
opts.kernel.y = y(k) + ny/2 - opts.center(2);

% dimensions of the data set
opts.dims = [nx ny nc nk];
if opts.loraks; opts.dims = [opts.dims 2]; end

% size of calibration matrix
bytes = 8*prod(opts.dims);
if isequal(opts.class,'double'); bytes = 2*bytes; end

% density of calibration matrix
density = nnz(mask) / numel(mask);

% display
t = tic; disp(opts); close;
figure('Position',[10 581 1792 425]);
fprintf('Sampling density = %f\n',density);
fprintf('Matrix = %ix%i (%.1f Mb)\n',nx*ny,prod(opts.dims(3:end)),bytes/1e6);

%% precision / gpu

data = cast(data,opts.class);
if opts.gpu
    gpu = gpuDevice;
    if gpu.AvailableMemory < 4*bytes || verLessThan('matlab','8.4')
        opts.gpu = 0;
        warning('need more GPU memory or MATLAB R2014b. Switching to CPU.')
    else
        data = gpuArray(data);
        mask = gpuArray(mask);
    end
end

%% POCS iterations - solve for ksp

keep = mask; % retained samples
ksp = zeros(nx,ny,nc,'like',data);

for iter = 1:opts.maxit
    
    % data consistency
    ksp = bsxfun(@times,data,keep)+bsxfun(@times,ksp,~keep);

    % make calibration matrix
    A = make_data_matrix(ksp,opts);

    % row space, singular vals, nuclear norm
    [~,S,V] = svd(A'*A,'econ');
    S = sqrt(diag(S));
    normA(iter) = sum(S);

    % initialize noise floor
    if isempty(opts.noise)
        
        % a "small" singular value
        for j = 1:numel(S)
            h = hist(S(j:end));
            [~,k] = max(h);
            if k>1; break; end
        end
        noise_floor = median(S(j:end));
        
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
    
    % geometric median (irls+huber)
    for k = 1:opts.irls_iters
        w = 1./abs(bsxfun(@minus,copy,ksp));
        w = min(w,1/opts.irls_scale/opts.noise);
        ksp = sum(w.*copy,4)./sum(w,4);
    end
    
    % tolerance ||Δk|| / ||k||
    if ~exist('old','var')
        tol(iter) = cast(NaN,'like',normA);
    else
        tol(iter) = norm(ksp(:)-old(:))/norm(ksp(:));
    end
    old = ksp;
    converged = tol(iter) < opts.tol;

    % residual (abs is less sensitive to translations)
    r = abs(data) - abs(ksp);
    %r = data - ksp;
    
    % project over coils
    r = sum(abs(r),3);
    
    % project over kspace dimension
    if opts.proj
        r(~keep) = 0;
        r = sum(r,opts.proj);
        index = find(any(keep,opts.proj));
    else    
        index = find(keep);
    end
    r = r(index);
    
    % median and std dev
    med_r = median(r);
    std_r = median(abs(r - med_r)) * 1.4826;

    % after convergence, trim the worst point
    if converged

        % no. std devs away from median
        [nstd k] = max(r/std_r-med_r/std_r);

        if nstd < opts.nstd
            rejected = 0; % reject nothing
        else
            rejected = index(k);
            if opts.proj==0; keep(rejected) = 0; end % reject worst point
            if opts.proj==1; keep(:,rejected) = 0; end % reject worst col
            if opts.proj==2; keep(rejected,:) = 0; end % reject worst row
        end

        % display info
        if ~exist('ntrim','var')
            ntrim = 0;
            disp('-----------------------------------------------------')
            disp(' Count   ||A||  Iteration Trimmed  nstd   std    Time');
        end
        ntrim = ntrim+1; % no. of trimmed points (or lines)
        fprintf('%5i %9.2e %6i',ntrim,normA(iter),iter);
        fprintf('%9i %6.1f %9.2e %4.0f\n',rejected,nstd,std_r,toc(t));

    end
    
    % display plots every few iterations
    if mod(iter,10)==1 || converged
        
        if iter==1
            % plot singular values
            subplot(1,4,1)
            plot(S/S(1));
            hold on; plot(max(f,min(ylim)),'--'); hold off
            line(xlim,gather([1 1]*noise_floor/S(1)),'linestyle',':','color','black');
            legend({'singular vals.','min. var. filter','noise floor'});
            title(sprintf('rank %i',nnz(f>0))); xlim([0 numel(S)+1]);
        else
            % show one kspace point
            subplot(1,4,1)
            coil = 1; x = opts.center(1)+1; y = opts.center(2)+1;
            plot(squeeze(data(x,y,coil)),'ogreen','MarkerFaceColor','green');
            hold on;
            plot(squeeze(copy(x,y,coil,:)),'o');
            plot(squeeze(mean(copy(x,y,coil,:))),'dred','MarkerFaceColor','red');
            plot(squeeze(ksp(x,y,coil)),'sblack','MarkerFaceColor','black');
            hold off
            title(sprintf('k-space point (%i,%i)',x,y)); xlabel('Re'); ylabel('Im');
            temp = copy(x,y,coil,:); temp = [real(temp(:)) imag(temp(:))];
            stddev(1) = median(abs(temp(:,1)-median(temp(:,1))));
            stddev(2) = median(abs(temp(:,2)-median(temp(:,2))));
            stddev = 1.4826 * mean(stddev);
            pos = [real(ksp(x,y,coil))-stddev imag(ksp(x,y,coil))-stddev 2*stddev 2*stddev];
            rectangle('Position',gather(pos),'Curvature',[1,1]); axis equal;
            legend({'data','copy','mean','huber'},'Location','SouthEast');
        end
        % show residual norm
        subplot(1,4,2);
        if opts.proj
            plot(index,r/std_r); title('residual projection'); ylabel('||r||_1  /  std dev');
            if opts.proj==1; xlim([0 ny+1]); xlabel('dim 2'); else xlim([0 nx+1]); xlabel('dim 1'); end
            line(xlim,[0 0]+med_r/std_r+opts.nstd,'linestyle','--','color','red');
            line(xlim,[0 0]+med_r/std_r-opts.nstd,'linestyle','--','color','red');
            line(xlim,[0 0]+med_r/std_r,'color','red'); axis tight;
        else
            temp = zeros(nx,ny,'like',r); temp(index) = r/std_r; ims(temp); 
            xlabel('dim 2'); ylabel('dim 1'); title('residual map'); colorbar
        end
        % show current image
        subplot(1,4,3);
        ims(sum(abs(ifft2(ksp)),3));
        xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iteration %i',iter));
        % plot change in norm
        subplot(1,4,4); warning('off','MATLAB:Axes:NegativeLimitsInLogAxis');
        [h,~,~] = plotyy(1:iter,max(tol,opts.tol),1:iter,normA);
        axis(h,'tight'); set(h(1),'YScale','log'); set(h(2),'YScale','log');
        title('convergence'); legend({'||Δk||/||k||','||A||_* norm'});
        xlim(h(1),[0 iter+1]); xlim(h(2),[0 iter+1]); xlabel('iteration');
        drawnow;

    end

    % finish when nothing left to do
    if converged && ~rejected; break; end

end

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
    B = flip(flip(A,1),2);
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
        B = conj(A(:,:,:,k+nk));
        B = flip(flip(B,2),1);
        A(:,:,:,k+nk) = circshift(B,-[x y]);
    end
end
