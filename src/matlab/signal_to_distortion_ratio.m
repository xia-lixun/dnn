function y = signal_to_distortion_ratio(x, t)
% x: signal to be evaluated
% t: reference signal (symbol)

    u = extract_symbol_and_merge(x, t);
    y = 10 * log10(sum(t.^2, 1) ./ sum((t-u).^2, 1));
end



function y = local_maxima(x)
% x must be colume vector
    gtl = [false; x(2:end) > x(1:end-1)];
    gtu = [x(1:end-1) >= x(2:end); false];
    y = gtl & gtu;
end


function y = extract_symbol_and_merge(x, s)
% x, s must be colume vector 
    m = length(s);
    if m > length(x)
        x = [x; zeros(m-length(x),1)];
    end
    n = length(x);
    y = zeros(m, 1);

    R = xcorr(s, x);
    Rs = sort(R(local_maxima(R)), 'descend');
    if isempty(Rs) 
        return
    end

    % find the anchor point
    ploc = find(R==Rs(1));
    lb = n - ploc + 1;
    rb = min(lb + m - 1, length(x));
    y(1:1+rb-lb) = x(lb:rb);
end
