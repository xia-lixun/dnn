function tensor(s, mu, std, flag)
    
    if strcmp(flag, 'training')
        n = s.training_examples;
    elseif strcmp(flag, 'testing')
        n = s.testing_examples;
    else
        error('flag = <training/testing>');
    end
    path_spectrum = fullfile(s.root, flag, 'spectrum');
    
    % prepare tensor folder
    path_tensor = fullfile(s.root, flag, 'tensor');
    if ~exist(path_tensor, 'dir')
        mkdir(path_tensor);
    end
    delete(fullfile(path_tensor, '*.mat'));

    % convert spectrum to tensor
    for i = 1:n
        load(fullfile(path_spectrum,['s_' num2str(i) '.mat']), 'bm', 'spec');
        spec = sliding((spec - mu)./std, (s.feature.context_span-1)/2, s.feature.nat_frames);
        save(fullfile(path_tensor,['t_' num2str(i) '.mat']), 'bm', 'spec');
    end
    
end



function y = symm(i,r)
    y = i-r:i+r;
end


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