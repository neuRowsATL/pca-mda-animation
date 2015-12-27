function pdat_labels = LabelData( pdat )
%LabelData categorizes projected data (# of classes determined by columns)
%   input: data in vector
%   output: returns 1 x length(data) array with labels

pdat_labels = zeros(1, size(pdat, 2));
nr_classes = size(pdat, 2);
class_1 = 0.35;
class_2 = 0.7;
class_3 = 1.0;

for pdat_row=1:size(pdat, 1)
    if pdat_row/length(pdat) <= class_1
        pdat_row/length(pdat);
        pdat_labels(pdat_row) = 1;
    elseif class_2 < pdat_row/length(pdat)
        pdat_row/length(pdat);
        pdat_labels(pdat_row) = 3;
    end
end

pdat_labels(pdat_labels == 0) = 2;

end

