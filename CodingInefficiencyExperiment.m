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

M = 2;
data = (1:M)'; %each sample is a row (like a database row)
epochs=10000;
lr = 0.1;

tic;
% alice = rand(1,M);
alice = [0.5, 0.5];
bob = [0.5, 0.5];
payoff_tracking = zeros(epochs,2);

for epoch=1:epochs
    %% Track payoffs
    payoffs = inefficiencyPayoffs(data, alice, bob);
    payoff_tracking(epoch,:) = payoffs;

    %% Calculate statistics
    jDist = jointDistribution(data, alice, bob);
    aDist = sum(jDist, 2);
    aBar = aDist(1);
    bDist = sum(jDist, 1);
    bBar = bDist(1);

    distACondB = jDist(:,1) / sum(jDist(:,1));
    pACondB = distACondB(1);
    distACondBc = jDist(:,2) / sum(jDist(:, 2));
    pACondBc = distACondBc(1);

    distBCondA = jDist(1,:) / sum(jDist(1,:));
    pBCondA = distBCondA(1);
    distBCondAc = jDist(2,:) / sum(jDist(2,:));
    pBCondAc = distBCondAc(1);

    %% Calculate samplewise conditionals
    pACondb_k = pACondB * bob' + pACondBc * (1-bob');
    pBConda_k = pBCondA * alice' + pBCondAc * (1-alice');

    
    %% Calculate Alice gradients
    %aGrad(k,i) = d pi_a_k / d a_i
    
    % Derivative of H(abar).
    dHofAbar = - ones(M,M) / M * (log(aBar) - log(1-aBar));
    
    % Derivative of L(a_k, abar)
    dLa_k__abar = -ones(M,M)/M .* (alice'./aBar - (1-alice')./(1-aBar));
    dLa_k__abar = dLa_k__abar - diag(log(aBar) - log(1-aBar) .* ones(1,M));

    % Derivative of H(P(A|b_k))
    dHPACondb_k = -ones(M,M)/M .* (log(pACondb_k) - log(1-pACondb_k)) .* ...
                    (bob' * bob ./ bBar + (1-bob') * (1-bob) ./ (1-bBar));

    % Derivative of L(a_k, P(A|b_k))
    dLa_k__PACondb_k = -ones(M,M)/M * (log(pACondb_k) - log(1-pACondb_k)) .* ...
                        (bob'  .* bob / bBar + (1-bob') .* (1-bob) / bBar);
    dLa_k__PACondb_k = dLa_k__PACondb_k - diag(log(pACondb_k) - log(1-pACondb_k));

    aGrads = dLa_k__abar - dHofAbar + dLa_k__PACondb_k - dHPACondb_k;
    ASampleGrads = ones(1,M) / M * aGrads;

    %% Calculate Bob gradients
    %bGrad(k,i) = d pi_b_k / d b_i
    
    % Derivative of H(bbar).
    dHofBbar = - ones(M,M) / M * (log(bBar) - log(1-bBar));
    
    % Derivative of L(b_k, bbar)
    dLb_k__bbar = -ones(M,M)/M .* (bob'./bBar - (1-bob')./(1-bBar));
    dLb_k__bbar = dLb_k__bbar - diag(log(bBar) - log(1-bBar) .* ones(1,M));

    % Derivative of H(P(B|a_k))
    dHPBConda_k = -ones(M,M)/M .* (log(pBConda_k) - log(1-pBConda_k)) .* ...
                    (alice' * alice ./ aBar + (1-alice') * (1-alice) ./ (1-aBar));

    % Derivative of L(b_k, P(B|a_k))
    dLb_k__PBConda_k = -ones(M,M)/M * (log(pBConda_k) - log(1-pBConda_k)) .* ...
                        (alice'  .* alice / aBar + (1-alice') .* (1-alice) / aBar);
    dLb_k__PBConda_k = dLb_k__PBConda_k - diag(log(pBConda_k) - log(1-pBConda_k));

    bGrads = dLb_k__bbar - dHofBbar + dLb_k__PBConda_k - dHPBConda_k;
    BSampleGrads = ones(1,M) / M * bGrads;

    %% Gradient ascend

%     alice = alice + lr * ASampleGrads;
    bob = bob + lr * BSampleGrads;
%     alice = min(max(alice, 0.001), 0.999);
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