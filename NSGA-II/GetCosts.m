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
%       Get ObjF                                                %
%===============================================================

function costs=GetCosts(pop)

    nobj=numel(pop(1).Cost);
    
    costs=reshape([pop.Cost],nobj,[]);

end