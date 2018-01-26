function y = sliding(x, r, t)
    
    [m,n] = size(x);
    head = repmat(x(:,1),1,r);
    tail = repmat(x(:,end),1,r);
    xhat = [head x tail];
    y = zeros((2*r+2)*m, n);
    
    for i = 1:n
        focus = xhat(:,symm(r+i,r));
        nat = mean(focus(:,1:t),2);
        y(:,i) = vec([focus nat]);
    end
end

function y = symm(i,r)
    y = i-r:i+r;
end