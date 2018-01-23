function y = cyclic_extend(x, n)
% x y belongs to 1-d array
    y = zeros(n,1);
    m = length(x);
    for i = 1:n
        y(i) = x(mod((i-1), m) + 1);
    end
end