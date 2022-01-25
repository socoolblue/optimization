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
%       Tournament selection                                    %
%         - Binari Tournament selection                         %
%===============================================================
                                                          
function [p,t]=Tournament(pop)     

    npop=numel(pop);
    i=randi([1 npop],1,2);
    p1=pop(i(1));
    p2=pop(i(2));

    if p1.Rank < p2.Rank
        p=p1;
        t=i(1);
    elseif p1.Rank > p2.Rank
        p=p2;
        t=i(2);
    else
        if p1.CrowdingDistance>p2.CrowdingDistance
            p=p1;
            t=i(1);
        else
            p=p2;
            t=i(2);
        end
    end

end