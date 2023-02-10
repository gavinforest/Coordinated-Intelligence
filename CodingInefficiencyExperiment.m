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

tic;
alice = rand(1,4);
bob = rand(1,4);


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

% Derivative of H(abar).
dHAbar = - ones(M,M) / M * (log(aBar) - log(1-aBar));

% Derivative of L(a_k, abar)
dLa_k__abar = -ones(M,M)/M .* (alice'./aBar - (1-alice')./(1-aBar));
dLa_k__abar = dLa_k__abar - diag(log(aBar) - log(1-aBar) .* ones(1,M));

% Derivative of L(a_k, P(A|b_k))
dLa_k__PACondb_k = -ones(M,M)/M .* (alice' ./ pACondb_k - (1-alice') ./ (1-pACondb_k)) .* ...
    (bob'  .* bob / bBar + (1-bob') .* (1-bob) / (1-bBar));
dLa_k__PACondb_k = dLa_k__PACondb_k - diag(log(pACondb_k) - log(1-pACondb_k));

% Derivative of H(P(A|b_k))
dHPACondb_k = -ones(M,M)/M .* (log(pACondb_k) - log(1-pACondb_k)) .* ...
    (bob' .* bob ./ bBar + (1-bob') .* (1-bob) ./ (1-bBar));

aGrads = dLa_k__abar - dHAbar + dLa_k__PACondb_k - dHPACondb_k;
ASampleGrads = ones(1,M) / M * aGrads;

%% Approximate Alice gradients with small perturbations
perturbationGrads = zeros(4, M, M);
ogInefficiencyTensor = inefficiencyTensor(jDist, data, alice, bob);
for s=1:M
    perturbation = zeros(size(alice));
    perturbation(s) = lr;
    perturbedAlice = alice + perturbation;

    perturbedJDist = jointDistribution(data, perturbedAlice, bob);
    perturbedIneffTensor = inefficiencyTensor(perturbedJDist, data, perturbedAlice, bob);


    perturbationGrads(:, :, s) = (perturbedIneffTensor(:,:,1) - ogInefficiencyTensor(:,:,1)) / lr;
end

approxDLa_k__abar= permute(perturbationGrads(1,:,:), [2 3 1]);
approxDHAbar = permute(perturbationGrads(2,:,:), [2 3 1]);
approxDLa_k__PACondb_k = permute(perturbationGrads(3,:,:), [2 3 1]);
approxDHPACondb_k = permute(perturbationGrads(4, :,:), [2 3 1]);



%% Local Functions
    
function [payoffs] = inefficiencyPayoffs(data, alice, bob)
    jDist = jointDistribution(data, alice, bob);
    inefficiencyTensory = inefficiencyTensor(jDist, data, alice, bob);
    signs = [1 -1 1 -1]';
    payoffs = mean(inefficiencyTensory .* signs, [1 2]);
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
    
        aliceIneffs = [L(pASamp, pA) H(pA) L(pASamp, pACondBk) H(pACondBk)];
        bobIneffs = [L(pBSamp, pB) H(pB) L(pBSamp, pBCondAk) H(pBCondAk)];
        
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