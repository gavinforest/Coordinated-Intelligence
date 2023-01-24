function [y] = linearClassify(x, params)
    output = params(1,:) .* x + params(2,:);
    y = output .* (output > 0);
    y = exp(y) ./ sum(exp(y), 'all');
end
