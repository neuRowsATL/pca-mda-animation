function [ output_args ] = HCAClass(data, numb_classes)
%HCAClass Classification pipeline for clustered data
%   input:  - data (vector of length N containing the projected data)
%           - numb_classes (# of classes to target)

% Label the data
if size(data, 1) < size(data, 2)
    pdat_labels = LSC(data', numb_classes);
elseif size(data, 1) > size(data, 2)
    pdat_labels = LSC(data, numb_classes);
end




end

