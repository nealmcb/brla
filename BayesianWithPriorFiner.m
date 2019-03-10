% Simulating the Bayesian Audit with a given prior and fine resolution. 
% Generating an audit (stopping) rule equivalent to the Bayesian audit
% Given an m-tier audit for large m, and m-vector n containing the m 
% specified audit sample sizes, this script generates an m-vector of 
% minimum number of votes for the winner required at each tier to stop 
% the audit. 

%------------------
% Input: 
%   N: votes cast for two candidates.
%   m: number of escalating audits
%   n: vector of size m containing audit sizes as audit escalates
%   NumberErrorValues: number of values of Bayesian error for which 
%                       stopping rules to be derived
%   BayesianError: vector of size NumberErrorValues containing acceptable 
%			Bayesian risk(s)

%----------
% Output:
%   kmin: array of minimum values of k; (i,j)th  value is the minimum 
%           number of votes for winner required to terminate jth tier of 
%           the audit derived for ith Bayesian risk.

N = 35295;
m=10000;
n = (1:m);

%NumberErrorValues = 7;
%BayesianError = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001);
NumberErrorValues = 1;
BayesianError = (0.05);

%-------------
% Computed values.
% HalfN: Maximum votes for announced winner for the election outcome to be 
%           incorrect. That is, half the total votes if N is even, and a 
%           losing margin of one if N is odd. 
% BaseLine: Weight of uniform prior
HalfN = floor(N/2);
BaseLine = 1/(N+1);

% ----------
% Computed Values 
% BayesianPrior: vector of size N+1 with values of the prior pdf for each 
%                       possible value of votes for the winner

% Uniform Prior
% BayesianPrior = (1/(N+1))*ones(1,N+1);

% Prior for alpha = beta = 1/2
% BayesianPrior = zeros(1,N+1);
% for r=1:N-1
%     BayesianPrior(r+1)=1/(sqrt((r/N)*(1-(r/N))));
% end
% BayesianPrior = BayesianPrior/sum(BayesianPrior);

% Prior for Bayesian RLA beginning with uniform
BayesianPrior = [zeros(1,HalfN), 0.5, (0.5/(N-HalfN))*ones(1,N-HalfN)];

%---------------------
% MODELLING THE BAYESIAN RISK
%   We estimate the Bayesian risk for the given prior. 
%   MATLAB's discrete beta distribution does not appear to be working too
%   well, so we explicitly compute it here. 

%   BayesPosteriorDist: computed posterior distribution; note that our 
%                           computation samples the hypergeometric pmf
%                           for different values of the true number of 
%                           votes for the winner, and is hence itself NOT
%                           normalized. We hence normalize it. 
%   k: number of votes for winner in sample
%   j: audit tier level
%   i: error level
%   Error: corresponding array of Bayesian risk values (sanity check, 
%           differs from prescribed values because k is integer, but error 
%           should not be larger than prescribed Bayesian risk). 

% For audit tier level j of size n(j), for each possible value of k 
% beginning at kmin(j-1), we determine the probability of error in 
% calling the winner for this value of k. We stop when the probability 
% of error is not larger than BayesianError. This value of k gives us the 
% stopping rule, kmin(j) for the corresponding value of n(j).

kprev = zeros(NumberErrorValues,m+1);
kmin = zeros(NumberErrorValues,m);
Error = zeros(NumberErrorValues,m);

for i=1:NumberErrorValues
    kprev(i,1)=1;
    for j=1:m
        for k=kprev(i,j):n(j)
            BayesPosteriorDist = BayesianPrior.*hygepdf(k,N,0:1:N,n(j));
            BayesPosteriorDist = BayesPosteriorDist/sum(BayesPosteriorDist);
            ThisError = sum(BayesPosteriorDist(1:HalfN+1));
            if ThisError <= BayesianError(i) 
                break
            end
        end
        kmin(i,j) = k;
        kprev(i,j+1)=k;
        Error(i,j) = ThisError;
    end
end
for i=1:NumberErrorValues
    for j=1:m
        if kmin(i,j)==n(j)
            kmin(i,j)=0;
            Error(i,j)=0;
        end
    end
end
kmin
Error
