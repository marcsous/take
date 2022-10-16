disp('Running example...')
clear all; close all;

%% simulation data
%load phantom0 % fully sampled
%load phantom1 % 2x undersampled
%load phantom2 % 2x undersampled
load phantom3 % 2x undersampled
figure; take2(data,'std',0.08,'errors',added_errors);

%% in vivo data
%load head1 % fully sampled
%load head2 % fully sampled
load head3 % fully sampled
figure; take2(data);
