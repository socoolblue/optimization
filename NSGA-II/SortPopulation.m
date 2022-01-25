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
%   Sorting with Rank and Crowding distance                     %
%   Create Population                                           %
%===============================================================
                                                         

function pop=SortPopulation(pop)

    CD=[pop.CrowdingDistance];
    [CD CD_sort_order]=sort(CD,'ascend');
    pop=pop(CD_sort_order);
    
    R=[pop.Rank];
    [R R_sort_order]=sort(R);
    pop=pop(R_sort_order);
    
end