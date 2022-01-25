%===============================================================
%                                                               %
%  MATLAB Code for Stepwise Opt.                                %
%  Non-dominated Sorting Genetic Algorithm II (NSGA-II)         %
%                                                               %
%                                                               %
%  Sejong Univ. K.-S. Sohn                                      %
%                                                               %
%         e-Mail: kssohn@sejong.ac.kr                           %
%         M.P:  010-6253-5913                                   %
%                                                               %
%         Create Empty Individuals                              %
%================================================================

function pop=CreateEmptyIndividuals(n)
clc;
    if nargin<1
        n=1;
    end
    
    individual.Position=[];
    individual.Cost=[];
    individual.Rank=[];
    individual.CrowdingDistance=[];
    individual.DominationSet=[];
    individual.DominatedCount=[];
    
    pop=repmat(individual,n,1);
    
end