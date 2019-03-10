% Simulating the Bayesian Audit with a given prior. 
% Generating an audit (stopping) rule equivalent to the Bayesian audit
% Given an m-tier audit with m-vector n containing the m specified audit 
% sample sizes, this script generates an m-vector of minimum number of 
% votes for the winner required at each tier to stop the audit. 

%------------------
% Input: 
%   N: votes cast for two candidates.
%   m: number of escalating audits
%   n: vector of size m containing audit sizes as audit escalates
%   NumberErrorValues: number of values of Bayesian error (upset probability) 
%			for which stopping rules to be derived
%   BayesianError: vector of size NumberErrorValues containing acceptable 
%			error(s) (upset probabilities) of the Bayesian Audit(s)
%   BayesianPrior: vector of size N+1 with values of the prior pdf for each 
%                       possible value of votes for the winner

N = 41700;
%m = 9;
%n = [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200];
m = 11;
n = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600];
% m=43;
% n = [25       30      36      43      52      62      75      90      107     129     155     186     223     267     321     385     462     555     666     799     958     1150        1380        1656        1987        2385        2862        3434        4121        4945        5934        7121        8546        10255       12306       14767       17720       21264       25517       30620       36744       44093       52912];

%m= 23;
%n = [25       35      49      69      134     188     264     369     517     723     1012        1417        1984        2778        3889        5445        7623        10672       14941       20917       29284     40997     57396];

Halfn = floor(n/2)+1;

%NumberErrorValues = 7;
%BayesianError = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];
%NumberErrorValues = 2;
%BayesianError = [0.1,0.005];
NumberErrorValues = 1;
BayesianError = [0.05];

%-------------
% Computed values.
% HalfN: Maximum votes for announced winner for the election outcome to be 
%           incorrect. That is, half the total votes if N is even, and a 
%           losing margin of one if N is odd. 
% BaseLine: Weight of uniform prior
HalfN = floor(N/2);
BaseLine = 1/(N+1);

%Uniform Prior
% BayesianPrior = (1/(N+1))*ones(1,N+1);

% Prior for alpha = beta = 1/2 for Bayesian Audit
% BayesianPrior = zeros(1,N+1);
% for r=1:N-1
%     BayesianPrior(r+1)=1/(sqrt((r/N)*(1-(r/N))));
% end
% BayesianPrior = BayesianPrior/sum(BayesianPrior);

% Prior for Bayesian RLA beginning with uniform
BayesianPrior = [zeros(1,HalfN), 0.5, (0.5/(N-HalfN))*ones(1,N-HalfN)];

%---------------------
% MODELLING THE ERROR OF THE BAYESIAN AUDIT
%   We estimate the error of the Bayesian Audit for the given prior. 
%   MATLAB's discrete beta distribution does not appear to be working too
%   well, so we explicitly compute it here. 

%   BayesPosteriorDist: computed posterior distribution; note that our 
%                           computation samples the hypergeometric pmf
%                           for different values of the true number of 
%                           votes for the winner, and is hence itself NOT
%                           normalized. We hence normalize it. 
%
%   k=number of votes for winner in sample
%   j=audit tier level
%   i=error level

% For audit tier level j of size n(j), for each possible value of k 
% beginning at n(j)/2 + 1, we determine the probability of error in 
% calling the winner for this value of k. We stop when the probability 
% of error is not larger than BayesianError (upset probability). This 
% value of k gives us the stopping rule for the corresponding value of 
% n(j). 

% kmin: array of minimum values of k; (i,j) value is the minimum number of 
%                votes for winner required to terminate jth tier of audit 
%                derived for ith audit measure.
% Error: corresponding array of Bayesian errors (sanity check, differs from 
%           prescribed values because k is integer, but should not be 
%           larger than prescribed values).  
kmin = zeros(NumberErrorValues,m);
Error = zeros(NumberErrorValues,m);

for i=1:NumberErrorValues
        for j=1:m
            for k=Halfn(j):n(j)
                BayesPosteriorDist = BayesianPrior.*hygepdf(k,N,0:1:N,n(j));
                BayesPosteriorDist = BayesPosteriorDist/sum(BayesPosteriorDist);
                ThisError = sum(BayesPosteriorDist(1:HalfN+1));
                if ThisError <= BayesianError(i) 
                    break
                end
            end
            kmin(i,j) = k;
            Error(i,j) = ThisError;
        end
end
kmin
Error
