%% Set up
data = [-1; 1]; %each sample is a row (like database row)

antiCoordGame = [0, 5; 3 1];
epochs = 5000;
lr = 0.01;
N = 250;
sim_payoff_tracking = zeros([N,2]);


%% Train
for simInd=1:N
    classifierA = rand([2,2]);
    classifierB = rand([2,2]);
    payoff_tracking = zeros([epochs 2]);
    for epoch=1:epochs
        for sampleInd=1:length(data)
            sample = data(sampleInd,:); %a ROW of data
            ys = [myclassify(sample, classifierA); myclassify(sample, classifierB)];
            payoffs = play(ys, antiCoordGame);
            payoff_tracking(epoch,:) = payoff_tracking(epoch,:) + payoffs'/size(data,1);
            classifierA = train(classifierA, sample, ys, 1, antiCoordGame,  lr);
            classifierB = train(classifierB, sample, ys, 2, antiCoordGame, lr);
        end
    end
    sim_payoff_tracking(simInd,:) = mean(payoff_tracking((epochs * 0.9):epochs,:),1);
end
%% Helper functions
function [classifier] = train(classifier, input, ys, outNum, game, lr)
    otherPlayerInd = mod(outNum, 2) + 1;
    dPayoffdSoftout = game * ys(otherPlayerInd,:)';
    dSoftoutdOutputs = ys(outNum,:) .* (eye(2,2) - ys(outNum,:));
    dOutputsdBs = 1;
    dOutputsdMs = input;

    dPayoffdBs = dOutputsdBs * dSoftoutdOutputs * dPayoffdSoftout;
    dPayoffdMs = dOutputsdMs .* (dSoftoutdOutputs * dPayoffdSoftout);

    classifier(1,:) = classifier(1,:) + lr * dPayoffdMs';
    classifier(2,:) = classifier(2,:) + lr * dPayoffdBs';
end

function [payoffs] = play(ys, game)
    
    payoffs = [ys(1,:) * game * ys(2,:)';
               ys(2,:) * game * ys(1,:)'];
end

function [y] = myclassify(x, params)
    output = params(1,:) .* x + params(2,:);
    y = output .* (output > 0);
    y = exp(y) ./ sum(exp(y), 'all');
end