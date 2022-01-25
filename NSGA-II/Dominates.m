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
%         Dominates Function                                    %
%================================================================
                                                           
function b=Dominates(p,q)

    if isstruct(p)
        p=p.Cost;
    end
    
    if isstruct(q)
        q=q.Cost;
    end

    b=(all(p>=q) && any(p>q));                             

end