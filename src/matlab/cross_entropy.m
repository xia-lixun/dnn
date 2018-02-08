function y = cross_entropy(t, x)
    % x: estimate of t, type = array{feature, frames}
    % t: target, type = array{feature, frames}
    y = mean2(t .* log(1./(x+eps)) + (1-t) .* log(1./(1-x+eps)));
end