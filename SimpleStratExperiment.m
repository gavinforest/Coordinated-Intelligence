%% Set up
data = (1:2)'; %each sample is a row (like database row)

antiCoordGame = [0 7; 5 0];
coordGame = [7 0; 0 5]; %note the imbalance!
epochs = 5000;
lr = 0.001;
N = 50;
sim_payoff_tracking = zeros([N,3]);
sim_sensitivity_tracking = zeros([N 3]);


%% Train
tic;
for simInd=1:N
    alice = rand([1 2]);
    bob = rand([1 2]);
    eve = rand([1 2]);
    payoff_tracking = zeros([epochs 3]);
    sensitivity_tracking = zeros([epochs 3]);
    numSamples = size(data, 1);
    for epoch=1:epochs
        aliceGrads = zeros([1 2]);
        bobGrads = zeros([1 2]);
        eveGrads = zeros([1 2]);
        ybars = zeros([3 2]);

        %Initial pass to establish ybars
        for sampleInd=1:numSamples
            s = data(sampleInd,:); %a ROW of data
            
            ys = [alice(s) (1- alice(s));
                  bob(s) (1- bob(s));
                  eve(s) (1-eve(s))];
            ybars = ybars + ys;
           
        end

        % Compute gradients pass
        for sampleInd=1:numSamples
            s = data(sampleInd, :);
            ys = [alice(s) (1- alice(s));
                  bob(s) (1- bob(s));
                  eve(s) (1-eve(s))];

            % alice and bob ascend anti-coordination game payoff
            aliceGrads(s) = aliceGrads(s) + [1 -1] * antiCoordGame * (ys(2,:) + ys(3,:))';
            bobGrads(s) = bobGrads(s) + [1 -1] * antiCoordGame * (ys(1,:) + ys(3,:))';
            % eve gradient ascends coordination game payoff
            eveGrads(s) = eveGrads(s) + [1 -1] * coordGame * (ys(1,:) + ys(2,:))';

            % Track game payoffs
            payoffs = [ys(1,:) * antiCoordGame * (ys(2,:)' + ys(3,:)'); %alice
                       ys(2,:) * antiCoordGame * (ys(1,:)' + ys(3,:)'); %bob
                       ys(3,:) * coordGame * (ys(1,:)' + ys(2,:)')]; %eve

            payoff_tracking(epoch,:) = payoff_tracking(epoch,:) + payoffs'/numSamples;
        end

        % Gradient ascend payoff
        alice = alice + lr * aliceGrads ./ numSamples;
        bob = bob + lr * bobGrads ./ numSamples;
        eve = eve + lr * eveGrads ./ numSamples;
        % Clamp to [0,1] probability range
        alice = min(max(alice, zeros('like', alice)), ones('like', alice));
        bob = min(max(bob, zeros('like', bob)), ones('like', bob));
        eve = min(max(eve, zeros('like', eve)), ones('like', eve));

        %Tracking statistics
        aliceSensitivity = norm(alice(1) - alice(2));
        bobSensitivity = norm(bob(1) - bob(2));
        eveSensitivity = norm(eve(1) - eve(2));
        sensitivity_tracking(epoch,:) = [aliceSensitivity bobSensitivity eveSensitivity];
    end
    sim_payoff_tracking(simInd,:) = mean(payoff_tracking((epochs * 0.9):epochs,:),1);
    sim_sensitivity_tracking(simInd, :) = mean(sensitivity_tracking((epochs * 0.9):epochs,:),1);
end
toc