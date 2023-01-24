function [grads] = computeGradients(input, ys, outNum, game, gradWeighting)
    otherPlayers = setdiff(1:3, outNum);
    weightedYs = gradWeighting .* ys;
    dPayoffdSoftout = game * sum(weightedYs(otherPlayers,:),1)';
    dSoftoutdOutputs = ys(outNum,:) .* (eye(2,2) - ys(outNum,:));
    dOutputsdBs = 1;
    dOutputsdMs = input;

    dPayoffdBs = dOutputsdBs * dSoftoutdOutputs * dPayoffdSoftout;
    dPayoffdMs = dOutputsdMs .* (dSoftoutdOutputs * dPayoffdSoftout);

    grads = [dPayoffdMs'; dPayoffdBs'];
end