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
%           Non-dominated Sorting (Pareto sorting)              %
%================================================================

function [pop F]=NonDominatedSorting(pop)

    F{1}=[];

    npop=numel(pop);
    for i=1:npop
        
        p=pop(i);
        p.DominationSet=[];
        p.DominatedCount=0;
        
        for j=1:npop
            if j==i
                continue;
            end
            
            q=pop(j);
            if Dominates(p,q)
                p.DominationSet=[p.DominationSet j];
            elseif Dominates(q,p)
                p.DominatedCount=p.DominatedCount+1;
            end
        end
        
        if p.DominatedCount==0
            p.Rank=1;
            F{1}=[F{1} i];
        end
        
        pop(i)=p;
        
    end
    
    f=1;
    while true
        
        Q=[];
        for i=1:numel(F{f})
            p=pop(F{f}(i));
            
            for j=1:numel(p.DominationSet)
                q=pop(p.DominationSet(j));
                
                q.DominatedCount=q.DominatedCount-1;
                if q.DominatedCount==0
                    q.Rank=f+1;
                    Q=[Q p.DominationSet(j)];
                end
                
                pop(p.DominationSet(j))=q;
            end
        end
        
        if isempty(Q)
            break;
        end
        
        F{f+1}=Q;
        f=f+1;
        
    end
    
    
end