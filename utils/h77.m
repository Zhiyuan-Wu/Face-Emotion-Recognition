function Feature = h77(I, method, dim)
%devide img into 7 by 7 sub-block, deploy method on them
%Input:
%	I: 64 by 64 img
%	method: function handle that can be used as method(I)
%	dim: length of feature that method return
%output:
%	Feature: a 1*(dim*49) feature
Features = zeros(49,dim);
count = 0;
for p = 1:7
    for q = 1:7
        count = count + 1;
        mask = I( 1+(p-1)*8:16+(p-1)*8 , 1+(q-1)*8:16+(q-1)*8 );
        Features(count,:) = method(mask);
    end
end
Feature = reshape(Features',[1,49*dim]);