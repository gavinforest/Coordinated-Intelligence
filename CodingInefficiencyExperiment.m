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

M = 4;
data = (1:M)'; %each sample is a row (like a database row)
lr = 0.001;
epochs = 1e4;
payoff_tracking = zeros(epochs, 2);

tic;
alice = rand(1,4);
bob = rand(1,4);

for epoch=1:epochs
    %% Track payoffs
    payoffs = inefficiencyPayoffs(data, alice, bob);
    payoff_tracking(epoch, :) = payoffs;


    %% Calculate statistics
    jDist = jointDistribution(data, alice, bob);
    aDist = jDist * [1;1];
    aBar = aDist(1);
    bDist = [1 1] * jDist;
    bBar = bDist(1);
    
    distACondB = jDist * [1;0];
    distACondB = distACondB / sum(distACondB);
    pACondB = distACondB(1);
    distACondBc = jDist * [0;1];
    distACondBc = distACondBc / sum(distACondBc);
    pACondBc = distACondBc(1);
    
    distBCondA = [1 0 ] * jDist;
    distBCondA = distBCondA / sum(distBCondA);
    pBCondA = distBCondA(1);
    distBCondAc = [0 1] * jDist;
    distBCondAc = distBCondAc / sum(distBCondAc);
    pBCondAc = distBCondAc(1);
    
    %% Calculate samplewise conditionals
    pACondb_k = pACondB * bob' + pACondBc * (1-bob');
    pBConda_k = pBCondA * alice' + pBCondAc * (1-alice');
    
    
    %% Calculate Alice gradients
    %aGrad(k,i) = d pi_a_k / d a_i
    
    % Derivative of H(a_k).
    dHa_k = - diag(log(alice) - log(1-alice));
    
    % Derivative of L(a_k, abar)
    dLa_k__abar = -ones(M,M)/M .* (alice'./aBar - (1-alice')./(1-aBar));
    dLa_k__abar = dLa_k__abar - diag(log(aBar) - log(1-aBar) .* ones(1,M));
    
    % Derivative of L(a_k, P(A|b_k))
    dLa_k__PACondb_k = -ones(M,M)/M .* (alice' ./ pACondb_k - (1-alice') ./ (1-pACondb_k)) .* ...
        (bob'  .* bob / bBar + (1-bob') .* (1-bob) / (1-bBar));
    dLa_k__PACondb_k = dLa_k__PACondb_k - diag(log(pACondb_k) - log(1-pACondb_k));
    
    
    aGrads = dLa_k__abar + dLa_k__PACondb_k - 2 * dHa_k;
    ASampleGrads = ones(1,M) / M * aGrads;

    %% Calculate Bob gradients
    %bGrad(k,i) = d pi_a_k / d a_i
    
    % Derivative of H(a_k).
    dHb_k = - diag(log(bob) - log(1-bob));
    
    % Derivative of L(a_k, abar)
    dLb_k__bbar = -ones(M,M)/M .* (bob'./bBar - (1-bob')./(1-bBar));
    dLb_k__bbar = dLb_k__bbar - diag(log(bBar) - log(1-bBar) .* ones(1,M));
    
    % Derivative of L(a_k, P(A|b_k))
    dLb_k__PBConda_k = -ones(M,M)/M .* (bob' ./ pBConda_k - (1-bob') ./ (1-pBConda_k)) .* ...
        (alice'  .* alice / aBar + (1-alice') .* (1-alice) / (1-aBar));
    dLb_k__PBConda_k = dLb_k__PBConda_k - diag(log(pBConda_k) - log(1-pBConda_k));
    
    
    bGrads = dLb_k__bbar + dLb_k__PBConda_k - 2 * dHb_k;
    BSampleGrads = ones(1,M) / M * bGrads;
    
    %% Gradient ascend payoffs
    alice = alice + lr * ASampleGrads + (rand(1,4) - 0.5) * lr;
    bob = bob + lr * BSampleGrads + (rand(1,4) - 0.5) * lr;
    alice = max(min(alice, 0.999), 0.001);
    bob = max(min(bob, 0.999), 0.001);
end



%% Local Functions
    
function [payoffs] = inefficiencyPayoffs(data, alice, bob)
    jDist = jointDistribution(data, alice, bob);
    ineffTensor = inefficiencyTensor(jDist, data, alice, bob);
    signs = [1 -1 1 -1]';
    payoffs = mean(ineffTensor .* signs, [1 2]) * 4;
end

function [ent] = L(a, p)
    ent = -(a .* log(p) + (1-a) .* log(1-p));
end

function [ent] = H(p)
    ent = -(p .* log(p) + (1-p) .* log(1-p));
end

function [inefficiencyTensor]= inefficiencyTensor(jDist, data, alice, bob)
    M = size(data, 1);
    inefficiencyTensor = zeros(4,M,2);

    pA = sum(jDist,2);
    pA = pA(1);
    pB = sum(jDist, 1);
    pB = pB(1);

    for sampleInd = 1:M
        s = data(sampleInd,:);
        pASamp = alice(s);
        pBSamp = bob(s);
    
        % Conditional probabilities
        pACondBk = jDist * [pBSamp; 1-pBSamp];
        pACondBk  = pACondBk  / sum(pACondBk);
        pACondBk = pACondBk(1);
        pBCondAk = [pASamp 1-pASamp] * jDist;
        pBCondAk = pBCondAk / sum(pBCondAk);
        pBCondAk = pBCondAk(1);
    
        aliceIneffs = [L(pASamp, pA) H(pASamp) L(pASamp, pACondBk) H(pASamp)];
        bobIneffs = [L(pBSamp, pB) H(pBSamp) L(pBSamp, pBCondAk) H(pBSamp)];
        
        inefficiencyTensor(:, sampleInd, 1) = aliceIneffs';
        inefficiencyTensor(:, sampleInd, 2) = bobIneffs';
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