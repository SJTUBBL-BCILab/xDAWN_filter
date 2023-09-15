% Data: 2023.08.14
% Author: Ruijie Luo
% Contact: luoruijie20@sjtu.edu.cn
% 
% Description: Implementation of xDAWN for ERP enhancement. Details can be found in the following article:
% Bertrand Rivet, Antoine Souloumiac, et al. xDAWN algorithm to enhance evoked potentials: application to brain computer interface.
% IEEE Transactions on Biomedical Engineering, 2009.
% 
% Implementation:
% Aim: Find the spatial filter applying xDAWN to enhance the signal to signal-plus-noise ratio and reduce feature dimension
% Assumption:
% Main idea:

classdef xDAWN_class < handle
    %xDAWN_class handbook:
    %1) Copy the xDAWN_class.m to the current work path or include the path of xDAWN_class.m

    %2) Initialize the object of xDAWN_class:
    % xDAWN_filter = xDAWN_class(raw_signal,trial_label,num_component,targetclass_label);
    % raw_signal(3D matrix): the epoched signal which is a 3D matrix [channel,time sample,trial]
    % trial_label(vector): the label for each trial
    % num_component(scalar): the number of selected xDAWN component
    % targetclass_label(scalar): the label value of target trials

    %3) train the spatial filter:
    % [spfed_signal, spatialfilter_weight] = spf_construct(xDAWN_filter);
    % spfed_signal(3D matrix): the spatial filtered training dataset [num_component,time sample,trial]
    % spatialfilter_weight(2D matrix): the weight of spatial filter [channel weight, component(same size as channel)]

    %4) spatial filter testing dataset:
    % spf_testdata = xDAWN_filter.spf_filtering(testdata);

    properties
        raw_signal;
        trial_label;
        num_component;
        targetclass_label;
        spatialfilter_weight;
    end

    properties(Access=private)
        num_channel;
        num_timesample;
        num_trial;
%         class_label;
        targetclass_index;
        nontargetclass_index;
    end

    methods
        function obj = xDAWN_class(raw_signal, trial_label, num_component, targetclass_label)
            %xDAWN Construct an instance of this class
            %   Detailed explanation goes here
            obj.raw_signal = raw_signal;
            obj.trial_label = trial_label;
            obj.num_component = num_component;
            obj.targetclass_label = targetclass_label;

            % Check input
            dimof_raw_signal = ndims(obj.raw_signal);
            if dimof_raw_signal~=3
                error('The input parameter raw_signal should be a 3D matrix');
            end
            if obj.num_component<=0
                error('The input parameter num_component should be a positive integer');
            end

            % Get channel, time, and trial from raw_signal
            [obj.num_channel, obj.num_timesample, obj.num_trial] = size(obj.raw_signal);
            if length(obj.trial_label)~=obj.num_trial
                error('The 3rd dimension of raw_signal should be equal to the length of trial_label');
            end
%             obj.class_label = unique(obj.trial_label);
            obj.targetclass_index = find(obj.trial_label==obj.targetclass_label);
            obj.nontargetclass_index = find(obj.trial_label~=obj.targetclass_label);
            if isempty(obj.targetclass_index)
                error('The label defined in targetclass_label is not in trial_label');
            end
        end

        function [spfed_signal, spatialfilter_weight] = spf_construct(obj)
            reshaped_raw_signal = reshape(obj.raw_signal, [obj.num_channel,obj.num_timesample*obj.num_trial]);
            reshaped_raw_signal = double(reshaped_raw_signal');

            % create Toeplitz matrix D
            stim_index_seq = 1:obj.num_timesample:(obj.num_timesample*obj.num_trial);
            stim_index_seq(obj.nontargetclass_index)=[];
            column_toeplitz = zeros(1,size(reshaped_raw_signal,1));
            column_toeplitz(stim_index_seq) = 1;
            row_toeplitz = zeros(1,obj.num_timesample);
            row_toeplitz(1) = column_toeplitz(1);
            D = toeplitz(column_toeplitz,row_toeplitz);
            D = sparse(D);

            % calculate the weight matrix of xDAWN
            [Qx,Rx] = qr(reshaped_raw_signal,'econ');
            [Qd,Rd] = qr(D,'econ');
            M = Qd'*Qx;
            [U,S,V] = svd(M,'econ');
            spatialfilter_weight = inv(Rx)*V;
            obj.spatialfilter_weight = spatialfilter_weight;
            filtered_signal = reshaped_raw_signal*obj.spatialfilter_weight;
            filtered_signal = filtered_signal(:,1:obj.num_component);
            spfed_signal = reshape(filtered_signal', [obj.num_component,obj.num_timesample,obj.num_trial]);
        end

        function spfed_signal = spf_filtering(obj,tobefiltered_signal)
            [tbf_channel, tbf_timesample, tbf_trial] = size(tobefiltered_signal);
            reshaped_raw_signal = reshape(tobefiltered_signal, [tbf_channel,tbf_timesample*tbf_trial]);
            if tbf_channel~=size(obj.spatialfilter_weight,1)
                error('The channel of tobefiltered_signal is not equal to the row of xDAWN weight matrix');
            end
            reshaped_raw_signal = double(reshaped_raw_signal');
            filtered_signal = reshaped_raw_signal*obj.spatialfilter_weight;
            filtered_signal = filtered_signal(:,1:obj.num_component);
            spfed_signal = reshape(filtered_signal', [obj.num_component,tbf_timesample,tbf_trial]);
        end
    end
end