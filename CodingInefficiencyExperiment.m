%% Experiment of Agents Learning Coding Inefficiency
% Alice and Bob each have one output in [0,1], viewed as a probability, and
% 4 samples upon which to vary their outputs. We view each output as the
% probabilty of a weighted coin, and each agent seeks to maximize the
% following
% 1. Given the average weight of my coins across all samples, maximize the
% average inefficiencies of encoding each weighted coin with that average
% 2. Given the conditional probability of a "heads" from alice/bob's coin
% given a "heads" from bob/alice's coin, and the output of bob/alice, what
% is the average coding inefficiency of alice/bob's weighted coin?

numSamples = 4;
data = (1:numSamples)'; %each sample is a row (like a database row)
epochs=10000;
lr = 0.001;

tic;
alice = rand(1,numSamples);
bob = rand(1,numSamples);
payoff_tracking = zeros(epochs,2);

for epoch=1:epochs
    %% Establish baseline payoffs
    payoffs = inefficiencyPayoffs(data, alice,bob);
    payoff_tracking(epoch,:) = payoffs;

    %% Approximate gradients for Alice
    aSteps = zeros(1,numSamples);
    for perturbInd=1:numSamples
        perturbation = zeros(1,numSamples);
        perturbation(perturbInd) = lr;
        perturbedPayoffs = inefficiencyPayoffs(data, alice + perturbation, bob);
        aSteps(perturbInd) = lr * (perturbedPayoffs(1) > payoffs(1)) ...
                            - lr * (perturbedPayoffs(1) < payoffs(1));
    end

    %% Approximatee gradients for Bob
    bSteps = zeros(1, numSamples);
    for perturbInd=1:numSamples
        perturbation = zeros(1,numSamples);
        perturbation(perturbInd) = lr;
        perturbedPayoffs = inefficiencyPayoffs(data, alice, bob + perturbation);
        bSteps(perturbInd) = lr * (perturbedPayoffs(2) > payoffs(2)) ...
                            - lr * (perturbedPayoffs(2) < payoffs(2));
    end

    %% Gradient ascend payoffs
    alice = alice + aSteps;
    bob = bob + bSteps;

    alice = min(max(alice, 0.001), 0.999);
    bob = min(max(bob, 0.001), 0.999);

end
    
function [payoffs] = inefficiencyPayoffs(data, alice, bob)
    jDist = jointDistribution(data, alice, bob);
    averageInefficiencies = inefficiencyMatrix(jDist, data, alice, bob);
    payoffs  =  sum(averageInefficiencies,2)';
end

function [averageInefficiencies]= inefficiencyMatrix(jDist, data, alice, bob)
    averageInefficiencies = zeros(2,2);
    % alice unconditional entropy
    pA = sum(jDist,2);
    pA = pA(1);
    % bob unconditional entropy
    pB = sum(jDist, 1);
    pB = pB(1);

    numSamples = size(data,1);
    for sampleInd = 1:numSamples
        s = data(sampleInd,:);
        pASamp = alice(s);
        pBSamp = bob(s);
    
        % Conditional probabilities
        pACondB = jDist * [pBSamp; 1-pBSamp];
        pACondB  = pACondB  / sum(pACondB);
        pACondB = pACondB(1);
        pBCondA = [pASamp 1-pASamp] * jDist;
        pBCondA = pBCondA / sum(pBCondA);
        pBCondA = pBCondA(1);
    
        sampleIneffs = [codingInefficiency(pASamp, pA) codingInefficiency(pASamp, pACondB);
                        codingInefficiency(pBSamp, pBCondA) codingInefficiency(pBSamp, pB)];
        
        averageInefficiencies = averageInefficiencies + sampleIneffs/numSamples;
    end

end

function [jDist] = jointDistribution(data, alice, bob)
    numSamples = size(data, 1);
    jDist = zeros(2,2);
    for sampleInd = 1:numSamples
        s = data(sampleInd,:);
        sampleJoint = [alice(s)*bob(s), alice(s)*(1-bob(s));
                             (1-alice(s))*bob(s), (1-alice(s))*(1-bob(s))];
        jDist = jDist  + sampleJoint / numSamples;
    end

end

function [ineff] = codingInefficiency(trueProb, codingProb)
    ineff = - (trueProb*logcl(codingProb) + (1-trueProb)*logcl(1-codingProb)) - binent(trueProb);
end

function [ent] = binent(p)
    if p == 0
        ent = 0;
        return;
    end
    ent = - (p.*logcl(p) + (1-p).*logcl(1-p));
end

function [y] = logcl(x)
    mask = x<=0;
    y = log2(x);
    y(mask) = 0;
end