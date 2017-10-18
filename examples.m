disp('Running example...')
clear all; close all;

%% simulation data
%load phantom0 % fully sampled
%load phantom1 % 2x undersampled
%load phantom2 % 2x undersampled
load phantom3 % 2x undersampled
fprintf('Added errors(%i) =',numel(added_errors)); fprintf(' %i',added_errors); fprintf('.\n');
figure; take2(data,mask);

%% in vivo data
%load head1 % fully sampled
%load head2 % fully sampled
load head3 % fully sampled
figure; take2(data,mask);
