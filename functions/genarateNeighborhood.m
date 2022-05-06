function indx_0 = genarateNeighborhood(KC,tau,DIRECTION)
if nargin<3
    DIRECTION = 'descend';
end
num = size(KC,1);
KC0 = KC - 10^8*eye(num);
[val,indx] = sort(KC0,DIRECTION);
indx_0 = indx(1:tau,:);