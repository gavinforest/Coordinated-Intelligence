%% Set up
data = [-1; 1]; %each sample is a row (like database row)

antiCoordGame = [0 7; 5 0];
coordGame = [2 0; 0 1]; %note the imbalance!
epochs = 4000;
lr = 0.01;
N = 50;
sim_payoff_tracking = zeros([N,3]);
sim_sensitivity_tracking = zeros([N 3]);


%% Train
tic;
parfor simInd=1:N
    alice = rand([2,2]);
    bob = rand([2,2]);
    eve = rand([2,2]);
    payoff_tracking = zeros([epochs 3]);
    sensitivity_tracking = zeros([epochs 3]);
    numSamples = size(data, 1);
    for epoch=1:epochs
        aliceGrads = zeros([2 2]);
        bobGrads = zeros([2 2]);
        eveGrads = zeros([2 2]);
        for sampleInd=1:length(data)
            sample = data(sampleInd,:); %a ROW of data
            ys = [linearClassify(sample, alice); 
                  linearClassify(sample, bob); 
                  linearClassify(sample, eve)];

            payoffs = playGames(ys, antiCoordGame, coordGame);
            payoff_tracking(epoch,:) = payoff_tracking(epoch,:) + payoffs'/numSamples;

            % How much weight each players action gets when computing grads
            pGradWeights = ones([3,2]);
            pGradWeights(3,:) = 0.5; 
            
            % alice and bob ascend anti-coordination game payoff
            aliceGrads = aliceGrads + computeGradients(sample, ys, 1, antiCoordGame, pGradWeights);
            bobGrads = bobGrads + computeGradients(sample, ys, 2, antiCoordGame, pGradWeights);
            % eve gradient ascends coordination game payoff
            eveGrads = eveGrads +  computeGradients(sample, ys, 3, coordGame, pGradWeights);
        end
        alice = alice + lr * aliceGrads ./ numSamples;
        bob = bob + lr * bobGrads ./ numSamples;
        eve = eve + lr * eveGrads ./ numSamples;
        aliceSensitivity = norm(linearClassify(1, alice) - linearClassify(-1, alice)) / sqrt(2);
        bobSensitivity = norm(linearClassify(1, bob) - linearClassify(-1, bob)) / sqrt(2);
        eveSensitivity = norm(linearClassify(1, eve) - linearClassify(-1, eve)) / sqrt(2);
        sensitivity_tracking(epoch,:) = [aliceSensitivity bobSensitivity eveSensitivity];
    end
    sim_payoff_tracking(simInd,:) = mean(payoff_tracking((epochs * 0.9):epochs,:),1);
    sim_sensitivity_tracking(simInd, :) = mean(sensitivity_tracking((epochs * 0.9):epochs,:),1);
end
toc